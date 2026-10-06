from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import torch
from utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib.ticker import ScalarFormatter
import gstlearn as gl
import time
import xarray as xr
import random





def load_data(path, times=range(100000, 100020), nb_bins=50, get_norm=False):

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
    # formatter_y = ScalarFormatter(useOffset=False)
    # formatter_y.set_scientific(False)
    # axs[0].yaxis.set_major_formatter(formatter_y)
    #
    # formatter_x = ScalarFormatter(useOffset=False)
    # formatter_x.set_scientific(False)
    # axs[0].xaxis.set_major_formatter(formatter_x)
    #
    # axs[0].ticklabel_format(style="plain", useOffset=False, axis="y")
    # axs[0].tick_params(axis="x", labelrotation=45)
    # axs[0].tick_params(axis="y", labelrotation=45)
    #
    # axs[1].imshow(
    #     res[0].T, aspect="equal", extent=[0, 1, 0, 1], origin="lower", cmap=CMAP
    # )
    #
    # # cbar = fig.colorbar(sc, ax=axs, location="right", shrink=1, pad=0.03)
    # # cbar.set_label("GHI")
    #
    # fig.suptitle("GHI 05:21:00 29 June 2013")
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

    # start_time = datetime.strptime("05:21:00 29 June 2013", "%H:%M:%S %d %B %Y")
    #
    # fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    #
    # # Premier subplot (Scatter)
    # sc = axs[0].scatter(east, north, c=vals[:, 0], cmap=CMAP)
    # axs[0].set_xlim(e_min, e_max)
    # axs[0].set_ylim(n_min, n_max)
    # axs[0].set_aspect("equal")
    # axs[0].set_xlabel("Easting (m)")
    # axs[0].set_ylabel("Northing (m)")
    # axs[0].set_title("Raw data")
    #
    # # Second subplot (Grid Projection)
    # im = axs[1].imshow(
    #     res[0].T, aspect="equal", extent=[0, 1, 0, 1], origin="lower", cmap=CMAP
    # )
    # axs[1].set_title("Grid Projection")
    #
    # # Colorbar globale
    # cbar = fig.colorbar(sc, ax=axs, location="right", shrink=1, pad=0.03)
    # cbar.set_label("GHI")
    #
    # title = fig.suptitle("")
    #
    # # Fonction de mise à jour (+1 minute par frame)
    # def update(frame):
    #     # Mise à jour des données graphiques
    #     sc.set_array(vals[:, frame])
    #     im.set_data(res[frame].T)
    #
    #     # Calcul et affichage de l'heure exacte (+1 min par frame)
    #     current_time = start_time + timedelta(minutes=frame)
    #     title.set_text(f"GHI {current_time.strftime('%H:%M:%S %d %B %Y')}")
    #
    #     return sc, im, title
    #
    # # Création et sauvegarde de l'animation
    # num_frames = len(vals)
    # anim = animation.FuncAnimation(
    #     fig, update, frames=num_frames, interval=200, blit=False
    # )
    #
    # output_path = Path("./output/results/spatiotemporal").resolve() / "paper/3D"
    # output_path.mkdir(parents=True, exist_ok=True)
    #
    # # 5 frames/sec = 5 minutes simulées par seconde dans le GIF
    # anim.save(output_path / "projection.gif", writer="pillow", fps=5)
    # plt.close()

    if get_norm :
        return res, ((vals * clear_sky[mask]).mean(), (vals * clear_sky[mask]).std())

    return res


    def get_gaussian_weighted_estimate(
            p_target: int,
            wide: int = 20,
            n_lags: int = 10,
            sigma: float = 3.0,
            space_size: int = space_size,
            output_quantiles: bool = True
    ):

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
                    preds_q05.append(torch.exp(quantile_CNN_q05(sample)[0]))
                    preds_q95.append(torch.exp(quantile_CNN_q95(sample)[0]))

        preds_median = torch.stack(preds_median)

        distances = torch.arange(-n_lags + 1, 1, dtype=torch.float32, device=preds_median.device)
        weights = torch.exp(-0.5 * (distances / sigma) ** 2)
        weights = weights / weights.sum()

        weighted_median = torch.sum(preds_median * weights.unsqueeze(1), dim=0)

        if output_quantiles:
            preds_q05 = torch.stack(preds_q05)
            preds_q95 = torch.stack(preds_q95)
            weighted_q05 = torch.sum(preds_q05 * weights.unsqueeze(1), dim=0)
            weighted_q95 = torch.sum(preds_q95 * weights.unsqueeze(1), dim=0)
            return weighted_median.cpu().numpy(), weighted_q05.cpu().numpy(), weighted_q95.cpu().numpy()

        return weighted_median.cpu().numpy()



def arrow(path, wind=None, times=range(100000, 100020)):
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

    # 1. Calcul des limites globales pour figer l'échelle de couleurs
    v_min, v_max = vals.min(), vals.max()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # start_time = datetime.strptime("05:21:00 29 June 2013", "%H:%M:%S %d %B %Y")
    # start_time = datetime.strptime("09:31:00 29 June 2013", "%H:%M:%S %d %B %Y")
    start_time = datetime.strptime("09:31:00 19 May 2013", "%H:%M:%S %d %B %Y")


    # 2. Initialisation du Scatter (avec vmin et vmax !)
    sc = ax.scatter(
        east,
        north,
        c=vals[:, 0],
        cmap=CMAP,
        vmin=v_min,
        vmax=v_max
    )
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Colorbar qui restera figée sur les vraies limites
    cbar = fig.colorbar(sc, ax=ax, location="right", shrink=1, pad=0.03)
    cbar.set_label("GHI")

    show_wind = not(wind is None)

    if show_wind :
        fig.suptitle("Wind direction over time")
    else :
        fig.suptitle("GHI over time")

    # Titre initial
    title = ax.set_title(f"{start_time.strftime('%H:%M:%S %d %B %Y')}")

    # Formatage des axes (défini une seule fois)
    formatter_y = ScalarFormatter(useOffset=False)
    formatter_y.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter_y)

    formatter_x = ScalarFormatter(useOffset=False)
    formatter_x.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter_x)

    ax.ticklabel_format(style="plain", useOffset=False, axis="y")
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=45)

    # Position statique du texte et du Quiver (flèche)
    x_pos = e_min + 0.75 * (e_max - e_min)
    y_pos = n_min + 0.80 * (n_max - n_min)

    if show_wind :
        ax.text(
            x_pos,
            y_pos + 0.003 * (n_max - n_min),
            "Wind",
            color="red",
            ha="center",
            weight="bold",
        )

        # Initialisation de la flèche (Quiver)
        u_wind_init = wind[0, 1].item()
        v_wind_init = wind[0, 0].item()
        q = ax.quiver(
            x_pos,
            y_pos,
            u_wind_init,
            v_wind_init,
            color="red",
            units="inches",
            scale=1,
            width=0.008,
            headwidth=4,
        )

    plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.88, wspace=0.3)

    # 3. Fonction de mise à jour propre
    def update(t):
        # Mise à jour des couleurs des points
        sc.set_array(vals[:, t])

        # Mise à jour du titre
        current_time = start_time + timedelta(minutes=t)
        title.set_text(f"{current_time.strftime('%H:%M:%S %d %B %Y')}")

        if show_wind:
            # Mise à jour de la flèche du vent (composantes U et V)
            u_wind = wind[t, 1].item()
            v_wind = wind[t, 0].item()
            q.set_UVC(u_wind, v_wind)
            return [sc, q, title]

        return [sc, title]

    anim = animation.FuncAnimation(fig, update, frames=len(times), interval=100)

    output_path = Path("./output/results/spatiotemporal").resolve() / "paper/3D"
    output_path.mkdir(parents=True, exist_ok=True)

    if show_wind :
        gif_path = output_path / f"wind_{times[0]}.gif"
    else :
        gif_path = output_path / f"no-wind_{times[0]}.gif"
    anim.save(gif_path, writer="pillow")

    plt.close(fig)
    
def sliding_window(
        output_path,
        start: int,
        end: int,
        CNN,
        quantile_CNN_low,
        quantile_CNN_high,
        space_size: int = 50,
        theta_names = (r"$v_1$", r"$v_2$", r"$\rho_t$", r"$\rho_s$", r"$\tau$")
):
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
    
                ax.plot(x_axis, res_wide[wide][:, k], label=f"Median of the posterior" if k == 0 else "", color=color, linewidth=.5)
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
            ax.set_xticks([start + 54, start + 60 * 4 + 54, start + 60 * 8 + 54, start + 60 * 12 + 54], labels=["6:00", "10:00", "14:00", "18:00"])
            ax.tick_params(axis='x', rotation=45)
            if k == 0 :
                ax.legend()
    
        # 3. Masquer le 6e sous-graphique inutile (ligne index 2, colonne index 1)
        axs[2, 1].set_visible(False)
    
        fig.suptitle(f"Parameters evolution during 29 June 2013")
        fig.tight_layout()
        plt.show()
    
        fig.savefig(output_path / f"sliding_window_wide.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(output_path / f"sliding_window_wide.png", format="png", bbox_inches="tight")

    # with torch.no_grad():
    #
    #     start_time = 104000
    #     end_time = 104840
    #
    #     theta_names = [r"$v_1$", r"$v_2$", r"$\rho_t$", r"$\rho_s$", r"$\tau$"]
    #
    #     fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    #
    #     res_wide = {20: torch.zeros((end_time - start_time, len(theta_names)))}
    #     res_q05 = {20: torch.zeros((end_time - start_time, len(theta_names)))}
    #     res_q95 = {20: torch.zeros((end_time - start_time, len(theta_names)))}
    #
    #     # Le compteur va maintenant accumuler des poids (des flottants) au lieu d'entiers
    #     counts = {20: torch.zeros((end_time - start_time, 1))}
    #
    #     stride = 1
    #     for i, wide in enumerate(res_wide.keys()):
    #         print(f"Processing wide={wide}")
    #
    #         # 1. CRÉATION DES POIDS GAUSSIENS (selon votre croquis)
    #         # sigma = 5, centrée au milieu de la fenêtre (9.5 pour une fenêtre de 20)
    #         x = torch.arange(wide, dtype=torch.float32)
    #         mu = (wide - 1) / 2.0
    #         sigma = 5.0
    #
    #         # Formule de la courbe en cloche : exp(-0.5 * ((x - mu) / sigma)^2)
    #         weights = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    #
    #         # On ajoute une dimension pour permettre le calcul matriciel avec les 5 paramètres
    #         # weights passe de shape (20,) à (20, 1)
    #         weights = weights.unsqueeze(1)
    #
    #         for p in range(start_time, end_time - wide + 1, stride):
    #             data = load_data(
    #                 "output/data/sun/HOPE_qcGHI_1min_v4.nc",
    #                 nb_bins=space_size,
    #                 times=range(p, p + wide),
    #             )
    #
    #             mask = torch.zeros_like(data, dtype=torch.bool)
    #             mask[torch.isnan(data)] = 1
    #             data[mask] = 0
    #
    #             sample = torch.stack([data, mask]).unsqueeze(0)
    #
    #             idx = torch.arange(p - start_time, p + wide - start_time)
    #
    #             # 2. ACCUMULATION PONDÉRÉE DES PRÉDICTIONS
    #             res_wide[wide][idx] += CNN(sample).squeeze() * weights
    #             res_q05[wide][idx] += torch.exp(quantile_CNN_q05(sample).squeeze()) * weights
    #             res_q95[wide][idx] += torch.exp(quantile_CNN_q95(sample).squeeze()) * weights
    #
    #             # 3. ACCUMULATION DES POIDS
    #             counts[wide][idx] += weights
    #
    #             if p % 100 == 0:
    #                 print(p, sample.shape)
    #
    #         print(counts[wide])
    #
    #         # 4. CALCUL DE LA MOYENNE PONDÉRÉE
    #         # On divise par la somme des poids (le clamp évite une division par 0 sur les bords non visités)
    #         valid_counts = counts[wide].clamp(min=1e-6)
    #
    #         res_wide[wide] /= valid_counts
    #         res_q05[wide] /= valid_counts
    #         res_q95[wide] /= valid_counts
    #
    #         x_axis = np.arange(start_time, end_time, stride)
    #         color = cols[[0, 2, 3, 5][i]]
    #
    #         # Indexation sur la grille 3x2 (k // 2 pour la ligne, k % 2 pour la colonne)
    #         for k, name in enumerate(theta_names):
    #             ax = axs[k // 2, k % 2]
    #
    #             ax.plot(x_axis, res_wide[wide][:, k], label=f"Median of the posterior" if k == 0 else "", color=color, linewidth=.5)
    #             ax.fill_between(
    #                 x_axis,
    #                 res_wide[wide][:, k] - res_q05[wide][:, k],
    #                 res_wide[wide][:, k] + res_q95[wide][:, k],
    #                 color=color,
    #                 alpha=0.2,
    #                 label=f"UQ 90%" if k == 0 else "",
    #                 linewidth=0.5,
    #             )
    #
    #     # Titres et cosmétique pour chaque paramètre
    #     for k, name in enumerate(theta_names):
    #         ax = axs[k // 2, k % 2]
    #         ax.set_title(name)
    #         ax.set_xticks([start_time + 54, start_time + 60 * 4 + 54, start_time + 60 * 8 + 54, start_time + 60 * 12 + 54], labels=["6:00", "10:00", "14:00", "18:00"])
    #         ax.tick_params(axis='x', rotation=45)
    #         if k == 0 :
    #             ax.legend()
    #
    #     # 3. Masquer le 6e sous-graphique inutile (ligne index 2, colonne index 1)
    #     axs[2, 1].set_visible(False)
    #
    #     fig.suptitle(f"Parameters evolution during 29 June 2013")
    #     fig.tight_layout()
    #     plt.show()
    #
    #     fig.savefig(output_path / f"sliding_window_wide_mean.pdf", format="pdf", bbox_inches="tight")
    #     fig.savefig(output_path / f"sliding_window_wide_mean.png", format="png", bbox_inches="tight")
    #
    #     arrow("output/data/sun/HOPE_qcGHI_1min_v4.nc", res_wide[20][:, :2], times=range(start_time, end_time))
    

def get_gaussian_weighted_estimate(
        p_target: int,
        CNN,
        quantile_CNN_low,
        quantile_CNN_high,
        wide: int = 20,
        n_lags: int = 10,
        sigma: float = 3.0,
        space_size: int = 50,
):
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

    distances = torch.arange(-n_lags + 1, 1, dtype=torch.float32, device=preds_median.device)
    weights = torch.exp(-0.5 * (distances / sigma) ** 2)
    weights = weights / weights.sum()

    weighted_median = torch.sum(preds_median * weights.unsqueeze(1), dim=0)

    if output_quantiles:
        preds_q05 = torch.stack(preds_q05)
        preds_q95 = torch.stack(preds_q95)
        weighted_q05 = torch.sum(preds_q05 * weights.unsqueeze(1), dim=0)
        weighted_q95 = torch.sum(preds_q95 * weights.unsqueeze(1), dim=0)
        return weighted_median.cpu().numpy(), weighted_q05.cpu().numpy(), weighted_q95.cpu().numpy()

    return weighted_median.cpu().numpy()