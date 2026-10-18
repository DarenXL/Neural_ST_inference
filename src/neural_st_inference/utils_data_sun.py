import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib.ticker import ScalarFormatter
import xarray as xr

from neural_st_inference.utils import *


def load_data(path, times=range(100000, 100020), nb_bins=50, get_norm=False):
    """
    Load, center and reduce sun data
    Project coordinates onto a grid
    """

    ds = xr.open_dataset(path, engine="netcdf4")

    north = ds.northing.values
    east = ds.easting.values
    vals = ds.qc_GHI.isel(time=times).values
    clear_sky = ds.cs_GHI.isel(time=times).values

    mask = ~np.isnan(vals).any(axis=1)
    vals = vals[mask] / clear_sky[mask]
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

    if get_norm:
        return res, ((vals * clear_sky[mask]).mean(), (vals * clear_sky[mask]).std())

    return res


def sliding_window(
    output_path,
    start: int,
    end: int,
    CNN,
    quantile_CNN_low,
    quantile_CNN_high,
    space_size: int = 50,
    theta_names=(r"$v_1$", r"$v_2$", r"$\rho_t$", r"$\rho_s$", r"$\sigma$", r"$\tau$"),
):
    """Apply a CNN over time"""
    with torch.no_grad():

        fig, axs = plt.subplots(3, 2, figsize=(12, 8))

        res_wide = {20: []}
        res_q05 = {20: []}
        res_q95 = {20: []}

        stride = 1
        for i, wide in enumerate(res_wide.keys()):
            print(wide)
            for p in range(start + wide, end, stride):
                data = load_data(
                    "output/data/sun/HOPE_qcGHI_1min_v4.nc",
                    nb_bins=space_size,
                    times=range(p - wide, p),
                )

                mask = torch.zeros_like(data, dtype=torch.bool)
                mask[torch.isnan(data)] = 1
                data[mask] = 0

                sample = torch.stack([data, mask]).unsqueeze(0)
                if p % 100 == 0:
                    print(p, sample.shape)

                res_wide[wide].append(CNN(sample)[0])
                res_q05[wide].append(torch.exp(quantile_CNN_low(sample)[0]))
                res_q95[wide].append(torch.exp(quantile_CNN_high(sample)[0]))

            res_wide[wide] = torch.stack(res_wide[wide]).cpu().numpy()
            res_q05[wide] = torch.stack(res_q05[wide]).cpu().numpy()
            res_q95[wide] = torch.stack(res_q95[wide]).cpu().numpy()

            x_axis = np.arange(start + wide, end, stride)
            color = cols[[0, 2, 3, 5][i]]

            # Indexation sur la grille 3x2 (k // 2 pour la ligne, k % 2 pour la colonne)
            for k, name in enumerate(theta_names):
                ax = axs[k // 2, k % 2]

                ax.plot(
                    x_axis,
                    res_wide[wide][:, k],
                    label=f"Median of the posterior" if k == 0 else "",
                    color=color,
                    linewidth=0.5,
                )
                ax.fill_between(
                    x_axis,
                    res_wide[wide][:, k] - res_q05[wide][:, k],
                    res_wide[wide][:, k] + res_q95[wide][:, k],
                    color=color,
                    alpha=0.2,
                    label=f"UQ 90%" if k == 0 else "",
                    linewidth=0.5,
                )

        # Titres et cosmétique pour chaque paramètre
        for k, name in enumerate(theta_names):
            ax = axs[k // 2, k % 2]
            ax.set_title(name)
            ax.set_xticks(
                [
                    start + 54,
                    start + 60 * 4 + 54,
                    start + 60 * 8 + 54,
                    start + 60 * 12 + 54,
                ],
                labels=["6:00", "10:00", "14:00", "18:00"],
            )
            ax.tick_params(axis="x", rotation=45)
            if k == 0:
                ax.legend()

        # 3. Masquer le 6e sous-graphique inutile (ligne index 2, colonne index 1)
        # axs[2, 1].set_visible(False)

        fig.suptitle(f"Parameters evolution during 29 June 2013")
        fig.tight_layout()
        plt.show()

        fig.savefig(
            output_path / f"sliding_window_wide.pdf", format="pdf", bbox_inches="tight"
        )
        fig.savefig(
            output_path / f"sliding_window_wide.png", format="png", bbox_inches="tight"
        )


def get_gaussian_weighted_estimate(
    p_target: int,
    CNN,
    quantile_CNN_low,
    quantile_CNN_high,
    wide: int = 20,
    n_lags: int = 10,
    sigma_window: float = 3.0,
    space_size: int = 50,
):
    """
    Used for forecasting, the last n_lag estimated parameters are ponderated with an half-Gaussian
    The last element has the biggest weigh
    """

    output_quantiles = True
    p_values = range(p_target - n_lags + 1, p_target + 1)

    preds_median = []
    preds_q05 = []
    preds_q95 = []

    with torch.no_grad():
        for p in p_values:
            data = load_data(
                "output/data/sun/HOPE_qcGHI_1min_v4.nc",
                nb_bins=space_size,
                times=range(p - wide, p),
            )

            mask = torch.zeros_like(data, dtype=torch.bool)
            mask[torch.isnan(data)] = 1
            data[mask] = 0.0

            sample = torch.stack([data, mask]).unsqueeze(0)

            preds_median.append(CNN(sample)[0])
            if output_quantiles:
                preds_q05.append(torch.exp(quantile_CNN_low(sample)[0]))
                preds_q95.append(torch.exp(quantile_CNN_high(sample)[0]))

    preds_median = torch.stack(preds_median)

    distances = torch.arange(
        -n_lags + 1, 1, dtype=torch.float32, device=preds_median.device
    )
    weights = torch.exp(-0.5 * (distances / sigma_window) ** 2)
    weights = weights / weights.sum()

    weighted_median = torch.sum(preds_median * weights.unsqueeze(1), dim=0)

    if output_quantiles:
        preds_q05 = torch.stack(preds_q05)
        preds_q95 = torch.stack(preds_q95)
        weighted_q05 = torch.sum(preds_q05 * weights.unsqueeze(1), dim=0)
        weighted_q95 = torch.sum(preds_q95 * weights.unsqueeze(1), dim=0)
        return (
            weighted_median.cpu().numpy(),
            weighted_q05.cpu().numpy(),
            weighted_q95.cpu().numpy(),
        )

    return weighted_median.cpu().numpy()
