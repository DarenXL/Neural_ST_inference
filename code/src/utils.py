from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


THETA_COLORS = [
    {
        "train": "#7AA7F1",  # "orangered",
        "test": "#7AA7F1",  # "lightsalmon",
        "baseline": "dimgrey",
    },
    {
        "train": "#648FFF",  # "deepskyblue",
        "test": "#648FFF",  # "lightskyblue",
        "baseline": "dimgrey",
    },
    {
        "train": "#785EF0",  # "forestgreen",
        "test": "#785EF0",  # "limegreen",
        "baseline": "dimgrey",
    },
    {
        "train": "#DC267F",  # "darkorchid",
        "test": "#DC267F",  # "orchid",
        "baseline": "dimgrey",
    },
    {
        "train": "#FE6100",
        "test": "#FE6100",
        "baseline": "dimgrey",
    },
    {
        "train": "#FFB000",
        "test": "#FFB000",
        "baseline": "dimgrey",
    },
]

def create_data_path(base_path: str) -> Path:
    path = Path(base_path).resolve()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = path / timestamp

    sub_dirs = [
        path / "train" / "samples",
        path / "test" # / "samples"
    ]

    for sd in sub_dirs:
        sd.mkdir(parents=True, exist_ok=True)

    return path

def create_res_path(base_path: str) -> Path:
    path = Path(base_path).resolve()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = path / timestamp

    sub_dirs = [
        path / "loss_evolution",
        path / "conclusion_graphs",
        path / "networks"
    ]

    for sd in sub_dirs:
        sd.mkdir(parents=True, exist_ok=True)

    return path


def sampling_matern(outShape, lambdaParent, lambdaDaughter, radiusCluster):
    xMin = 0
    xMax = 1
    yMin = 0
    yMax = 1

    rExt = radiusCluster
    xMinExt = xMin - rExt
    xMaxExt = xMax + rExt
    yMinExt = yMin - rExt
    yMaxExt = yMax + rExt

    xDeltaExt = xMaxExt - xMinExt
    yDeltaExt = yMaxExt - yMinExt
    areaTotalExt = xDeltaExt * yDeltaExt

    numbPointsParent = np.random.poisson(areaTotalExt * lambdaParent)

    xxParent = xMinExt + xDeltaExt * np.random.uniform(0, 1, numbPointsParent)
    yyParent = yMinExt + yDeltaExt * np.random.uniform(0, 1, numbPointsParent)

    numbPointsDaughter = np.random.poisson(lambdaDaughter, numbPointsParent)
    numbPoints = sum(numbPointsDaughter)

    theta = 2 * np.pi * np.random.uniform(0, 1, numbPoints)
    rho = radiusCluster * np.sqrt(np.random.uniform(0, 1, numbPoints))
    xx0 = rho * np.cos(theta)
    yy0 = rho * np.sin(theta)

    xx = np.repeat(xxParent, numbPointsDaughter)
    yy = np.repeat(yyParent, numbPointsDaughter)

    xx = xx + xx0
    yy = yy + yy0

    booleInside = ((xx >= xMin) & (xx <= xMax) & (yy >= yMin) & (yy <= yMax))
    xx = xx[booleInside]
    yy = yy[booleInside]

    # plt.scatter(xx,yy)
    # plt.xlim(xMin, xMax)
    # plt.ylim(yMin, yMax)
    # plt.axis('equal')
    # plt.title(f"{numbPointsParent} parents | {numbPointsDaughter} daughters")
    # plt.show()

    bin_size = 1.0 / outShape

    ix = np.clip((xx / bin_size).astype(np.int32), max=outShape - 1)
    iy = np.clip((yy / bin_size).astype(np.int32), max=outShape - 1)

    # flat_idx = ix * outShape + iy

    mask = np.zeros((outShape, outShape), dtype=bool)
    mask[ix, iy] = 1

    return mask

def add_missing_mask(batch, missing_ratio):
    deepset_size = batch.shape[0]
    space_size = batch.shape[-1]

    nb_missing = int(space_size * missing_ratio)
    nb_visible = space_size - nb_missing

    mask = np.ones_like(batch)

    for i in range(deepset_size):
        if len(batch.shape) == 4:
            if nb_visible < (space_size + 1) // 2:
                max_index = space_size - (nb_visible - 1)
                base = np.random.choice(max_index, size=nb_visible, replace=False)
                base = np.sort(base)
                visibles = base + np.arange(nb_visible)
            else:
                visibles = np.random.choice(space_size, size=nb_visible, replace=False)
        else :
            visibles = sampling_matern(space_size, nb_visible,nb_visible, .1)

        mask[i, :, :, visibles] = 0

    return np.concatenate([batch, mask], axis=1)

def deepset_collate(batch):
    data, theta, deepset = map(torch.stack, zip(*batch))

    data = data.flatten(0, 1)
    theta = theta.flatten(0, 1)
    deepset = deepset.flatten(0, 1)

    data[:, 0][data[:, 1] == 1] = 0

    return data, theta, deepset

def show_example(dataset, data_path, k=0, label='example'):

    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(18, 7), layout='constrained')

    if dataset.space_dim == 1:

        fig.suptitle(r"Examples on different $\theta$")

        for i in range(len(axs[0])):
            dataset.show(axs[0, i], k + i)
            dataset.show(axs[1, i], k + i, mask=True)

            axs[1, i].set_xlabel(r"$\it{Space}$")
            axs[1, i].set_ylabel(r"$\it{Time}$" if (i == 0) else None)
            axs[0, i].set_ylabel(r"$\it{Time}$" if (i == 0) else None)

    else:
        fig.suptitle(r"Example over time")

        dataset.show(axs[0], k)
        dataset.show(axs[1], k, mask=True)

        for i in range(len(axs[0])):
            # data_t = data[k, 0, i].clone()
            # data_t[data_t == 0] = np.nan
            # axs[1, i].imshow(data_t, aspect='equal', extent=[0,1,0,1])
            axs[1, i].set_xlabel(r"$\it{Space 1}$")
            axs[1, i].set_ylabel(r"$\it{Space 2}$")
            axs[0, i].set_ylabel(r"$\it{Space 2}$" if i == 0 else None)

    all_images = [im for ax in axs.flat for im in ax.get_images()]

    if all_images:
        v_min = min(np.nanmin(im.get_array()) for im in all_images)
        v_max = max(np.nanmax(im.get_array()) for im in all_images)

        cols = ["#7AA7F1","#648FFF","#785EF0","#DC267F","#FE6100","#FFB000"]

        cmap = LinearSegmentedColormap.from_list("", cols, N=100)

        for im in all_images:
            im.set_clim(v_min, v_max)
            im.set_cmap(cmap)

        cbar = fig.colorbar(all_images[-1], ax=axs, location='right', shrink=0.8)
        cbar.set_label('') # 'Value'

    plt.show()

    (data_path / "examples").mkdir(parents=True, exist_ok=True)
    fig.savefig(data_path / "examples" / f"{label}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(data_path / "examples" / f"{label}.png", format="png", bbox_inches="tight")

def unique_preserve_order(x):
    seen = []
    out = []
    for v in x.tolist():
        if v not in seen:
            seen.append(v)
            out.append(v)
    return torch.tensor(out, device=x.device)


def map_deepset(deepset, theta, device):
    unique_ds = unique_preserve_order(deepset)

    deepset_mapped = torch.tensor(
        [unique_ds.tolist().index(int(ds)) for ds in deepset],
        device=device
    )

    theta = torch.stack([theta[deepset == v][0] for v in unique_ds])

    return deepset_mapped, theta


def baseline_mae(dataset):
    all_thetas = dataset.get_theta_tensor()

    theta_mean = all_thetas.mean(dim=0)

    preds = theta_mean.unsqueeze(0).repeat(len(dataset), 1)

    mae = torch.mean(torch.abs(preds - all_thetas), dim=0)

    return mae, theta_mean

def global_mean_pool(x, deepset):
    num_groups = deepset.max().item() + 1
    out = torch.zeros(num_groups, x.size(1), device=x.device)
    out.index_add_(0, deepset, x)
    return out / torch.bincount(deepset).float().unsqueeze(1)

def format_seconds(s):
    h, m, s = int(s//3600), int((s%3600)//60), int(s%60)
    return f"{h}h{m}m{s}s" if h > 0 else (f"{m}m{s}s" if m > 0 else f"{s}s")
