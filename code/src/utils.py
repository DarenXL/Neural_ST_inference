from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib.ticker import ScalarFormatter
import gstlearn as gl
import time
import xarray as xr


THETA_COLORS = [
    {
        "train": "#7AA7F1",
        "test": "#7AA7F1",
        "baseline": "dimgrey",
    },
    {
        "train": "#648FFF",
        "test": "#648FFF",
        "baseline": "dimgrey",
    },
    {
        "train": "#785EF0",
        "test": "#785EF0",
        "baseline": "dimgrey",
    },
    {
        "train": "#c82072",
        "test": "#c82072",
        "baseline": "dimgrey",
    },
    {
        "train": "#DC267F",
        "test": "#DC267F",
        "baseline": "dimgrey",
    },
    {
        "train": "#e768a6",
        "test": "#e768a6",
        "baseline": "dimgrey",
    },
    # {
    #     "train": "#FE6100",
    #     "test": "#FE6100",
    #     "baseline": "dimgrey",
    # },
    # {
    #     "train": "#FFB000",
    #     "test": "#FFB000",
    #     "baseline": "dimgrey",
    # },
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
):
    """
    Poisson Sampling
    - Chooses a number of parents according to Poisson distribution (lambdaParent)
    - Places the parents in space
    - Chooses a number of daughter per parent according to Poisson distribution (lambdaDaughter)
    - Places the daughters around their parent (radiusCluster)
    """

    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0

    r_ext = radius_cluster
    x_min_ext = x_min - r_ext
    x_max_ext = x_max + r_ext
    y_min_ext = y_min - r_ext
    y_max_ext = y_max + r_ext

    x_delta_ext = x_max_ext - x_min_ext
    y_delta_ext = y_max_ext - y_min_ext
    area_total_ext = x_delta_ext * y_delta_ext

    numb_points_parent = torch.clamp(
        torch.poisson(torch.tensor([area_total_ext * lambda_parent])).int(), min=1
    ).item()

    xx_parent = x_min_ext + x_delta_ext * torch.rand(numb_points_parent)
    yy_parent = y_min_ext + y_delta_ext * torch.rand(numb_points_parent)

    numb_points_daughter = torch.clamp(
        torch.poisson(torch.full((numb_points_parent,), float(lambda_daughter))).int(),
        min=1,
    )
    numb_points = torch.sum(numb_points_daughter).item()

    theta = 2 * torch.pi * torch.rand(numb_points)
    rho = radius_cluster * torch.sqrt(torch.rand(numb_points))
    xx0 = rho * torch.cos(theta)
    yy0 = rho * torch.sin(theta)

    xx = torch.repeat_interleave(xx_parent, numb_points_daughter)
    yy = torch.repeat_interleave(yy_parent, numb_points_daughter)

    xx = xx + xx0
    yy = yy + yy0

    boole_inside = (xx >= x_min) & (xx <= x_max) & (yy >= y_min) & (yy <= y_max)
    xx = xx[boole_inside]
    yy = yy[boole_inside]

    bin_size = 1.0 / out_shape

    ix = torch.clamp((xx / bin_size).to(torch.int64), max=out_shape - 1)
    iy = torch.clamp((yy / bin_size).to(torch.int64), max=out_shape - 1)

    mask = torch.zeros((out_shape, out_shape), dtype=torch.bool)
    mask[ix, iy] = True

    if torch.all(~mask):
        mask[
            torch.randint(0, out_shape, (1,)).item(),
            torch.randint(0, out_shape, (1,)).item(),
        ] = True

    return mask


def sampling_poisson_1D(
    out_shape: int, lambda_parent: float, lambda_daughter: float, radius_cluster: float
):
    x_min = 0.0
    x_max = 1.0

    r_ext = radius_cluster
    x_min_ext = x_min - r_ext
    x_max_ext = x_max + r_ext

    x_delta_ext = x_max_ext - x_min_ext

    numb_points_parent = torch.clamp(
        torch.poisson(torch.tensor([x_delta_ext * lambda_parent])).int(), min=1
    ).item()

    xx_parent = x_min_ext + x_delta_ext * torch.rand(numb_points_parent)

    numb_points_daughter = torch.clamp(
        torch.poisson(torch.full((numb_points_parent,), float(lambda_daughter))).int(), min=1
    )
    numb_points = torch.sum(numb_points_daughter).item()

    theta = 2 * torch.pi * torch.rand(numb_points)
    rho = radius_cluster * torch.sqrt(torch.rand(numb_points))
    xx0 = rho * torch.cos(theta)

    xx = torch.repeat_interleave(xx_parent, numb_points_daughter)

    xx = xx + xx0

    boole_inside = (xx >= x_min) & (xx <= x_max)
    xx = xx[boole_inside]

    bin_size = 1.0 / out_shape

    ix = torch.clamp((xx / bin_size).to(torch.int64), max=out_shape - 1)

    mask = torch.zeros((out_shape,), dtype=torch.bool)
    mask[ix] = True

    if torch.all(~mask):
        mask[torch.randint(0, len(mask), (1,)).item()] = True

    return mask


def create_mask(
    space_size: int, space_shape: int, missing_ratio: float = None
):
    """Create a mask based on Poisson Sampling"""

    if missing_ratio is None:
        # missing_ratio = torch.rand(1) * 0.35 + 0.6   # space 30
        missing_ratio = torch.rand(1) * 0.08 + 0.9   # space 100
        # missing_ratio = torch.rand(1) * 0.25 + 0.7  # space 75x75

    nb_missing = int(space_size * missing_ratio)
    nb_visible = space_size - nb_missing

    if space_shape == 1:
        visibles = sampling_poisson_1D(space_size, nb_visible, nb_visible, 0.1)
    else:
        visibles = sampling_poisson_2D(space_size, nb_visible, nb_visible, 0.1)

    return visibles


def show_mask(outshape, dim=1, lambda_p=(3, 3, 12, 12), lambda_d=(3, 12, 3, 12)):

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(14, 5), layout="constrained")
    CMAP = LinearSegmentedColormap.from_list("custom_data", [cols[3], cols[3]], N=100)

    if dim == 2:
        for ax, lp, ld in zip(axs, lambda_p, lambda_d):
            sampling = sampling_poisson_2D(outshape, lp, ld, 0.1)
            res = torch.zeros_like(sampling, dtype=torch.float)
            res[~sampling] = torch.nan
            ax.imshow(res, cmap=CMAP, extent=[0, 1, 0, 1])
            ax.set_title(r"$\lambda_p = $" + str(lp) + r", $\lambda_d =$" + str(ld))
    else:
        for ax, lp, ld in zip(axs, lambda_p, lambda_d):
            sampling = sampling_poisson_1D(outshape, lp, ld, 0.1)
            res = torch.zeros((outshape, outshape), dtype=torch.float)
            res[:, ~sampling] = torch.nan
            ax.imshow(res, cmap=CMAP, extent=[0, 1, 0, 1])
            ax.set_title(r"$\lambda_p = $" + str(lp) + r", $\lambda_d =$" + str(ld))

    fig.savefig("mask_poisson.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def collate(batch):
    """used in the loader to merge batch"""
    zipped = list(zip(*batch))

    data = torch.cat(zipped[0], dim=0)
    theta = torch.cat(zipped[1], dim=0)

    data[:, 0][data[:, 1] == 1] = 0

    if len(zipped) == 3:
        classi = torch.cat(zipped[2], dim=0)
        return data, theta, classi

    return data, theta


def show_example(dataset, data_path: Path, k=0, label="example"):
    """Plot examples of gaussian fields"""

    fig, axs = plt.subplots(
        nrows=1, ncols=4, figsize=(10, 3.5), layout="constrained", squeeze=False
    )

    if dataset.space_dim == 1:

        fig.suptitle(r"Examples on different $\theta$")

        for i in range(len(axs[0])):
            # dataset.show(axs[0, i], k + i)
            dataset.show(axs[0, i], k + i, mask=None)

            # axs[1, i].set_xlabel(r"$\it{Space}$")
            # axs[1, i].set_ylabel(r"$\it{Time}$" if (i == 0) else None)
            axs[0, i].set_ylabel(r"Time" if (i == 0) else None)
            axs[0, i].set_xlabel(r"Space")

    else:
        fig.suptitle(r"Example over time")

        # dataset.show(axs[1], k, mask=True)

        for i in range(len(axs[0])):

            dataset.show(axs[0, i], k, mask=None, time_idx=i)

            # axs[1, i].set_xlabel(r"$\it{Space 1}$")
            # axs[1, i].set_ylabel(r"$\it{Space 2}$")
            axs[0, i].set_ylabel(r"Space 2" if i == 0 else None)
            axs[0, i].set_xlabel(r"Space 1")

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
            all_images[-1], ax=axs, location="right", shrink=1.1, pad=0.025
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


def show_example_mask(dataset, data_path: Path, k=0, label="example", mask=None):
    """Plot examples of gaussian fields and their masked apearance"""

    fig, axs = plt.subplots(
        nrows=2, ncols=4, figsize=(10, 5), layout="constrained", squeeze=False
    )

    if dataset.space_dim == 1:

        fig.suptitle(r"Examples on different $\theta$")

        for i in range(len(axs[0])):

            mask = dataset.get_mask(k+i)

            dataset.show(axs[0, i], k + i)
            dataset.show(axs[1, i], k + i, mask=mask)

            axs[1, i].set_xlabel(r"Space")
            axs[1, i].set_ylabel(r"Time" if (i == 0) else None)
            axs[0, i].set_ylabel(r"Time" if (i == 0) else None)

    else:
        fig.suptitle(r"Example over time")

        if mask is None:
            mask = dataset.get_mask(k)

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
            all_images[-1], ax=axs, location="right", shrink=1.05, pad=0.025
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

    v_min, v_max = dataset.get_min_max(np.arange(k, k + len(axs[0])))

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

    anim = animation.FuncAnimation(fig, update, frames=num_time_steps, interval=200)

    out_dir = data_path / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    gif_path = out_dir / f"{label}_animated.gif"

    anim.save(gif_path, writer="pillow")

    plt.close(fig)

    print(gif_path)


def load_data(path, times=range(100000, 100025), nb_bins=75):

    ds = xr.open_dataset(path, engine="netcdf4")

    north = ds.northing.values
    east = ds.easting.values
    vals = ds.raw_GHI.isel(time=times).values

    mask = ~np.isnan(vals).any(axis=1)
    vals = vals[mask]
    north = north[mask]
    east = east[mask]

    n_min, n_max = north.min(), north.max()
    e_min, e_max = east.min(), east.max()

    e_span = e_max - e_min
    n_span = n_max - n_min
    max_span = max(e_span, n_span)

    if e_span < n_span:
        e_max = e_min + max_span
    else:
        n_max = n_min + max_span

    north_norm = (north - n_min) / (n_max - n_min)
    east_norm = (east - e_min) / (e_max - e_min)
    vals_norm = (vals - vals.mean()) / (vals.std())

    y = torch.from_numpy(north_norm)
    x = torch.from_numpy(east_norm)
    values = torch.from_numpy(vals_norm).T

    bin_size = 1.0 / nb_bins

    ix = torch.clamp((x / bin_size).long(), max=nb_bins - 1)
    iy = torch.clamp((y / bin_size).long(), max=nb_bins - 1)

    flat_idx = ix * nb_bins + iy

    value_flat = torch.full((values.shape[0], nb_bins * nb_bins), float("nan"))

    unique_idx = flat_idx.unique()
    for idx in unique_idx:
        indices_in_cell = torch.where(flat_idx == idx)[0]
        chosen = indices_in_cell[torch.randint(0, len(indices_in_cell), (1,))]
        value_flat[:, idx] = values[:, chosen.squeeze()]

    res = value_flat.view(values.shape[0], nb_bins, nb_bins)


    # fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    #
    # for ax in axs:
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)
    #     ax.set_aspect("equal")
    #
    # sc = axs[0].scatter(east, north, c=vals[:, 0], cmap=CMAP,)
    # axs[0].set_xlim(e_min, e_max)
    # axs[0].set_ylim(n_min, n_max)
    # axs[0].set_xlabel("Easting (m)")
    # axs[0].set_ylabel("Northing (m)")
    #
    # axs[1].imshow(
    #     res[0].T, aspect="equal", extent=[0, 1, 0, 1], origin="lower", cmap=CMAP
    # )
    #
    # cbar = fig.colorbar(sc, ax=axs, location="right", shrink=1, pad=0.03)
    # cbar.set_label("GHI")
    #
    # fig.suptitle("GHI 04:01:00 6th June 2013")
    # axs[0].set_title("Raw data")
    # axs[1].set_title("Grid Projection")
    #
    # plt.show()
    # plt.tight_layout()
    #
    # output_path = Path("./output/results/spatiotemporal").resolve() / f"paper/3D"
    #
    # fig.savefig(
    #     output_path / f"projection.pdf", format="pdf", bbox_inches="tight"
    # )
    # fig.savefig(output_path / f"projection.png", format="png", bbox_inches="tight")

    return res

def arrow(path, wind, times=range(100000, 100025)):

    ds = xr.open_dataset(path, engine="netcdf4")

    north = ds.northing.values
    east = ds.easting.values
    vals = ds.raw_GHI.isel(time=times).values

    mask = ~np.isnan(vals).any(axis=1)
    vals = vals[mask]
    north = north[mask]
    east = east[mask]

    n_min, n_max = north.min(), north.max()
    e_min, e_max = east.min(), east.max()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    sc = ax.scatter(east, north, c=vals[:, 0], cmap=CMAP,)
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    cbar = fig.colorbar(sc, ax=ax, location="right", shrink=1, pad=0.03)
    cbar.set_label("GHI")

    fig.suptitle("Wind direction on time")

    # Pour l'axe Y (Northing)
    formatter_y = ScalarFormatter(useOffset=False)
    formatter_y.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter_y)

    # Pour l'axe X (Easting) si besoin
    formatter_x = ScalarFormatter(useOffset=False)
    formatter_x.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter_x)

    def update(t):
        ax.clear()

        ax.scatter(
            east,
            north,
            c=vals[:, t],
            cmap=CMAP,
        )
        ax.set_xlim(e_min, e_max)
        ax.set_ylim(n_min, n_max)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(f"Time {t}")

        u_wind = wind[t, 1].item()
        v_wind = wind[t, 0].item()

        x_pos = e_min + 0.80 * (e_max - e_min)
        y_pos = n_min + 0.80 * (n_max - n_min)

        ax.quiver(
            x_pos,
            y_pos,
            u_wind,
            v_wind,
            color="red",
            units="inches",  # La taille dépend de pouces physiques
            scale=20,
            width=0.008,
            headwidth=4,
        )

        ax.text(
            x_pos,
            y_pos + 0.003 * (n_max - n_min),
            "Wind",
            color="red",
            ha="center",
            weight="bold",
        )

        ax.ticklabel_format(style="plain", useOffset=False, axis="y")
        ax.tick_params(axis="x", labelrotation=45)
        ax.tick_params(axis="y", labelrotation=45)

        return [ax.get_images()]

    plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.88, wspace=0.3)

    anim = animation.FuncAnimation(fig, update, frames=len(times), interval=100)

    output_path = Path("./output/results/spatiotemporal").resolve() / f"paper/3D"
    output_path.mkdir(parents=True, exist_ok=True)

    gif_path = output_path / f"wind.gif"

    anim.save(gif_path, writer="pillow")


def baseline_mae(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a baseline based on the generated theta"""
    all_thetas = dataset.get_theta_tensor()

    theta_mean = all_thetas.mean(dim=0)

    preds = theta_mean.unsqueeze(0).repeat(dataset.theta_size, 1)

    mae = torch.mean(torch.abs(preds - all_thetas), dim=0)

    return mae, theta_mean


def format_seconds(s: float) -> str:
    """Times in second to h-m-s format"""
    h, m, s = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h}h{m}m{s}s" if h > 0 else (f"{m}m{s}s" if m > 0 else f"{s}s")

def test_load_time(loader, network, device):
    # Test sur 10 itérations
    train_iter = iter(loader)

    t_data, t_gpu_transfer, t_forward, t_backward = 0, 0, 0, 0
    MAE = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    for _ in range(10):
        # 1.
        t0 = time.perf_counter()
        batch = next(train_iter)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_data += t1 - t0

        # 2.
        data, theta = (
            batch[0].cuda(),
            batch[1].cuda(),
        )
        data = data.to(device, non_blocking=True).float()
        theta = theta.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_gpu_transfer += t2 - t1

        # 3.
        optimizer.zero_grad()
        outputs = network(data)
        loss = MAE(outputs, theta)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_forward += t3 - t2

        # 4.
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        t_backward += t4 - t3

    print(f"Mean per batch :")
    print(f"- Lecture DataLoader : {t_data / 10:.4f}s")
    print(f"- Transfer toward GPU : {t_gpu_transfer / 10:.4f}s")
    print(f"- Forward : {t_forward / 10:.4f}s")
    print(f"- Backward : {t_backward / 10:.4f}s")
