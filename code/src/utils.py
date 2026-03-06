from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

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

cols = [c["train"] for c in THETA_COLORS]
CMAP = LinearSegmentedColormap.from_list("custom_data", cols, N=100)


def create_data_path(base_path: str) -> Path:
    """Create a new path for save data"""

    path = Path(base_path).resolve()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = path / timestamp

    sub_dirs = [path / "train" / "samples", path / "test"]  # / "samples"

    for sd in sub_dirs:
        sd.mkdir(parents=True, exist_ok=True)

    return path


def create_res_path(base_path: str) -> Path:
    """Create a new path for save results"""

    path = Path(base_path).resolve()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = path / timestamp

    sub_dirs = [path / "loss_evolution", path / "conclusion_graphs", path / "networks"]

    for sd in sub_dirs:
        sd.mkdir(parents=True, exist_ok=True)

    return path


def sampling_poisson_2D(
    out_shape: int, lambda_parent: float, lambda_daughter: float, radius_cluster: float
) -> np.ndarray:
    """
    Poisson Sampling
    - Chooses a number of parents according to Poisson distribution (lambdaParent)
    - Places the parents in space
    - Chooses a number of daughter per parent according to Poisson distribution (lambdaDaughter)
    - Places the daughters around their parent (radiusCluster)
    """

    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1

    r_ext = radius_cluster
    x_min_ext = x_min - r_ext
    x_max_ext = x_max + r_ext
    y_min_ext = y_min - r_ext
    y_max_ext = y_max + r_ext

    x_delta_ext = x_max_ext - x_min_ext
    y_delta_ext = y_max_ext - y_min_ext
    area_total_ext = x_delta_ext * y_delta_ext

    numb_points_parent = np.maximum(
        np.random.poisson(area_total_ext * lambda_parent), 1
    )

    xx_parent = x_min_ext + x_delta_ext * np.random.uniform(0, 1, numb_points_parent)
    yy_parent = y_min_ext + y_delta_ext * np.random.uniform(0, 1, numb_points_parent)

    numb_points_daughter = np.maximum(
        np.random.poisson(lambda_daughter, numb_points_parent), 1
    )
    numb_points = sum(numb_points_daughter)

    theta = 2 * np.pi * np.random.uniform(0, 1, numb_points)
    rho = radius_cluster * np.sqrt(np.random.uniform(0, 1, numb_points))
    xx0 = rho * np.cos(theta)
    yy0 = rho * np.sin(theta)

    xx = np.repeat(xx_parent, numb_points_daughter)
    yy = np.repeat(yy_parent, numb_points_daughter)

    xx = xx + xx0
    yy = yy + yy0

    boole_inside = (xx >= x_min) & (xx <= x_max) & (yy >= y_min) & (yy <= y_max)
    xx = xx[boole_inside]
    yy = yy[boole_inside]

    bin_size = 1.0 / out_shape

    ix = np.clip((xx / bin_size).astype(np.int32), max=out_shape - 1)
    iy = np.clip((yy / bin_size).astype(np.int32), max=out_shape - 1)

    mask = np.zeros((out_shape, out_shape), dtype=bool)
    mask[ix, iy] = 1

    return mask


def sampling_poisson_1D(
    out_shape: int, lambda_parent: float, lambda_daughter: float, radius_cluster: float
) -> np.ndarray:
    x_min = 0
    x_max = 1

    r_ext = radius_cluster
    x_min_ext = x_min - r_ext
    x_max_ext = x_max + r_ext

    x_delta_ext = x_max_ext - x_min_ext

    numb_points_parent = np.maximum(np.random.poisson(x_delta_ext * lambda_parent), 1)

    xx_parent = x_min_ext + x_delta_ext * np.random.uniform(0, 1, numb_points_parent)

    numb_points_daughter = np.maximum(
        np.random.poisson(lambda_daughter, numb_points_parent), 1
    )
    numb_points = sum(numb_points_daughter)

    theta = 2 * np.pi * np.random.uniform(0, 1, numb_points)
    rho = radius_cluster * np.sqrt(np.random.uniform(0, 1, numb_points))
    xx0 = rho * np.cos(theta)

    xx = np.repeat(xx_parent, numb_points_daughter)

    xx = xx + xx0

    boole_inside = (xx >= x_min) & (xx <= x_max)
    xx = xx[boole_inside]

    bin_size = 1.0 / out_shape

    ix = np.clip((xx / bin_size).astype(np.int32), max=out_shape - 1)

    mask = np.zeros((out_shape,), dtype=bool)
    mask[ix] = 1

    return mask

def create_mask(space_size:int, space_shape:int, missing_ratio: float = None) -> np.ndarray:
    """Create a mask based on Poisson Sampling"""

    if missing_ratio is None:
        missing_ratio = np.random.random() * 0.35 + 0.6

    nb_missing = int(space_size * missing_ratio)
    nb_visible = space_size - nb_missing

    if space_shape == 1:
        # if nb_visible < (space_size + 1) // 2:
        #     max_index = space_size - (nb_visible - 1)
        #     base = np.random.choice(max_index, size=nb_visible, replace=False)
        #     base = np.sort(base)
        #     visibles = base + np.arange(nb_visible)
        # else:
        #     visibles = np.random.choice(space_size, size=nb_visible, replace=False)

        visibles = sampling_poisson_1D(space_size, nb_visible, nb_visible, 0.1)
        # visibles = sampling_poisson_2D(space_size, nb_visible, nb_visible, 0.1)
    else:
        visibles = sampling_poisson_2D(space_size, nb_visible, nb_visible, 0.1)

    return visibles

def add_missing_mask(batch: np.ndarray, missing_ratio: float = None) -> np.ndarray:
    """Concatenate each element of a batch with a mask based on Poisson Sampling"""
    deepset_size = batch.shape[0]
    space_size = batch.shape[-1]

    mask = np.ones_like(batch)

    for i in range(deepset_size):

        mask[i, :, :, create_mask(space_size, 1 if len(batch.shape) == 4 else 2, missing_ratio)] = 0

    return np.concatenate([batch, mask], axis=1)


def deepset_collate(batch: np.ndarray):
    """used in the loader to merge batch and deepset dimensions"""
    data, theta, deepset = map(torch.stack, zip(*batch))

    data = data.flatten(0, 1)
    theta = theta.flatten(0, 1)
    deepset = deepset.flatten(0, 1)

    data[:, 0][data[:, 1] == 1] = 0

    return data, theta, deepset


def show_example(dataset, data_path: Path, k=0, label="example"):
    """Plot examples of gaussian fields"""

    fig, axs = plt.subplots(
        nrows=1, ncols=4, figsize=(14, 5), layout="constrained", squeeze=False
    )

    if dataset.space_dim == 1:

        fig.suptitle(r"Examples on different $\theta$")

        for i in range(len(axs[0])):
            # dataset.show(axs[0, i], k + i)
            dataset.show(axs[0, i], k + i, mask=None)

            # axs[1, i].set_xlabel(r"$\it{Space}$")
            # axs[1, i].set_ylabel(r"$\it{Time}$" if (i == 0) else None)
            axs[0, i].set_ylabel(r"$\it{Time}$" if (i == 0) else None)
            axs[0, i].set_xlabel(r"$\it{Space}$")

    else:
        fig.suptitle(r"Example over time")

        # dataset.show(axs[1], k, mask=True)

        for i in range(len(axs[0])):

            dataset.show(axs[0, i], k, mask=None, time_idx=i)

            # axs[1, i].set_xlabel(r"$\it{Space 1}$")
            # axs[1, i].set_ylabel(r"$\it{Space 2}$")
            axs[0, i].set_ylabel(r"$\it{Space 2}$" if i == 0 else None)
            axs[0, i].set_xlabel(r"$\it{Space 1}$")

    all_images = [im for ax in axs.flat for im in ax.get_images()]

    if all_images:
        v_min = min(np.nanmin(im.get_array()) for im in all_images)
        v_max = max(np.nanmax(im.get_array()) for im in all_images)

        # cols = ["#7AA7F1","#648FFF","#785EF0","#DC267F","#FE6100","#FFB000"]
        # cmap = LinearSegmentedColormap.from_list("", cols, N=100)

        for im in all_images:
            im.set_clim(v_min, v_max)
            im.set_cmap(CMAP)

        cbar = fig.colorbar(
            all_images[-1], ax=axs, location="right", shrink=1.2, pad=0.025
        )
        cbar.set_label("")  # 'Value'

    plt.show()

    (data_path / "examples").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        data_path / "examples" / f"{label}.pdf", format="pdf", bbox_inches="tight"
    )
    fig.savefig(
        data_path / "examples" / f"{label}.png", format="png", bbox_inches="tight"
    )


def show_example_mask(dataset, data_path: Path, k=0, label="example"):
    """Plot examples of gaussian fields and their masked apearance"""

    fig, axs = plt.subplots(
        nrows=2, ncols=4, figsize=(14, 8), layout="constrained", squeeze=False
    )

    if dataset.space_dim == 1:

        fig.suptitle(r"Examples on different $\theta$")

        for i in range(len(axs[0])):
            mask = create_mask(
                dataset.space_size, dataset.space_dim,
                missing_ratio=dataset.missing_ratio,
            )

            dataset.show(axs[0, i], k + i)
            dataset.show(axs[1, i], k + i, mask=mask)

            axs[1, i].set_xlabel(r"$\it{Space}$")
            axs[1, i].set_ylabel(r"$\it{Time}$" if (i == 0) else None)
            axs[0, i].set_ylabel(r"$\it{Time}$" if (i == 0) else None)

    else:
        fig.suptitle(r"Example over time")

        mask = create_mask(
                dataset.space_size, dataset.space_dim,
                missing_ratio=dataset.missing_ratio,
            )

        for i in range(len(axs[0])):

            dataset.show(axs[0, i], k, time_idx=i)
            dataset.show(axs[1, i], k, mask=mask, time_idx=i)

            axs[1, i].set_xlabel(r"$\it{Space 1}$")
            axs[1, i].set_ylabel(r"$\it{Space 2}$")
            axs[0, i].set_ylabel(r"$\it{Space 2}$" if i == 0 else None)

    all_images = [im for ax in axs.flat for im in ax.get_images()]

    if all_images:
        v_min = min(np.nanmin(im.get_array()) for im in all_images)
        v_max = max(np.nanmax(im.get_array()) for im in all_images)

        for im in all_images:
            im.set_clim(v_min, v_max)
            im.set_cmap(CMAP)

        cbar = fig.colorbar(
            all_images[-1], ax=axs, location="right", shrink=1.2, pad=0.025
        )
        cbar.set_label("")  # 'Value'

    plt.show()

    (data_path / "examples").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        data_path / "examples" / f"{label}_mask.pdf", format="pdf", bbox_inches="tight"
    )
    fig.savefig(
        data_path / "examples" / f"{label}_mask.png", format="png", bbox_inches="tight"
    )


def show_example_gif(
    dataset, data_path: Path, num_time_steps: int, k=0, label="example"
):

    fig, axs = plt.subplots(
        nrows=2, ncols=4, figsize=(14, 8), layout="constrained", squeeze=False
    )
    fig.suptitle(r"Examples on different $\theta$")

    masks = []
    for _ in range(len(axs[0])):
        mask = create_mask(
            dataset.space_size,
            dataset.space_dim,
            missing_ratio=dataset.missing_ratio,
        )
        masks.append(mask)

    cbar = None

    v_min, v_max = dataset.get_min_max(np.arange(k, k+len(axs[0])))

    dataset.show(axs[0, 0], k, time_idx=0)
    sample_im = axs[0, 0].get_images()[0]
    sample_im.set_clim(v_min, v_max)
    sample_im.set_cmap(CMAP)

    cbar = fig.colorbar(sample_im, ax=axs, location="right", shrink=1.25, pad=0.03)
    cbar.set_label("")

    def update(t):
        for i in range(len(axs[0])):
            axs[0, i].clear()
            axs[1, i].clear()

            dataset.show(axs[0, i], k + i, time_idx=t)
            dataset.show(axs[1, i], k + i, mask=masks[i], time_idx=t)

            for im in axs[0, i].get_images() + axs[1, i].get_images():
                im.set_clim(v_min, v_max)
                im.set_cmap(CMAP)

            axs[1, i].set_xlabel(r"$\it{Space 1}$")
            if i == 0:
                axs[0, i].set_ylabel(r"$\it{Space 2}$")
                axs[1, i].set_ylabel(r"$\it{Space 2}$")

        return [ax.get_images() for ax in axs.flat]

    anim = animation.FuncAnimation(fig, update, frames=num_time_steps, interval=500)

    out_dir = data_path / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    gif_path = out_dir / f"{label}_animated.gif"

    anim.save(gif_path, writer="pillow")

    plt.close(fig)

    print(gif_path)


def unique_preserve_order(x: torch.Tensor) -> torch.Tensor:
    """remove all duplicates without changing the order"""
    seen = []
    out = []
    for v in x.tolist():
        if v not in seen:
            seen.append(v)
            out.append(v)
    return torch.tensor(out, device=x.device)


def map_deepset(
    deepset: torch.Tensor, theta: torch.Tensor, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjusts the shape of theta to be equal to the output of the neural network after deepset"""
    unique_ds = unique_preserve_order(deepset)

    deepset_mapped = torch.tensor(
        [unique_ds.tolist().index(int(ds)) for ds in deepset], device=device
    )

    theta = torch.stack([theta[deepset == v][0] for v in unique_ds])

    return deepset_mapped, theta


def baseline_mae(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a baseline based on the generated theta"""
    all_thetas = dataset.get_theta_tensor()

    theta_mean = all_thetas.mean(dim=0)

    preds = theta_mean.unsqueeze(0).repeat(dataset.theta_size, 1)

    mae = torch.mean(torch.abs(preds - all_thetas), dim=0)

    return mae, theta_mean


def global_mean_pool(x: torch.Tensor, deepset: torch.Tensor) -> torch.Tensor:
    """Apply deepset on summary statistics"""
    num_groups = deepset.max().item() + 1
    out = torch.zeros(num_groups, x.size(1), device=x.device)
    out.index_add_(0, deepset, x)
    return out / torch.bincount(deepset).float().unsqueeze(1)


def format_seconds(s: float) -> str:
    """Times in second to h-m-s format"""
    h, m, s = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h}h{m}m{s}s" if h > 0 else (f"{m}m{s}s" if m > 0 else f"{s}s")
