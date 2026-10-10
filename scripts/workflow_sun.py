import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import gstlearn.document as gdoc
import copy

gdoc.setNoScroll()

import matplotlib.dates as mdates
from neural_st_inference.utils import *
from neural_st_inference.utils_data_sun import *
from neural_st_inference.models.gneiting_adv import GneitingAdvModel
from neural_st_inference.models.gneiting_sep import GneitingSepModel
from neural_st_inference.data.datasets import GenericSimulationDataset, NlsSimulationDataset
from neural_st_inference.architecture.train_test import *
from neural_st_inference.mle.mle_test import *
from neural_st_inference.architecture.neural_networks import CNN_ST_network, print_parameters_per_layer
from neural_st_inference.architecture.output_graphs import *
import warnings
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        # "font.family": "inconsolata"
    }
)

#########################
####### Variables #######
#########################

# res_path = create_res_path("output/results/spatiotemporal")
res_path = Path("./output/results/spatiotemporal").resolve() / "2026-09-24_16-50-46_adv-3D"

output_path = Path("./output/results/spatiotemporal").resolve() / f"paper/3D"

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Neural Network train options

batch_size = 50
nb_epochs = 500
freq_epochs_test = 10

# Data options

time_size = 20
space_dim = 2
space_size = 50

theta_names = [r"$v_1$", r"$v_2$", r"$\rho_t$", r"$\rho_s$", r"$\tau$"]

if __name__ == "__main__":

    ##############################
    ##############################
    ############ Data ############
    ##############################
    ##############################

    ####################
    ####### Load #######
    ####################

    model = GneitingAdvModel(space_dim=space_dim, noise=.5)
    ini = 104120 # 104700 # 104120 # 104000 # range(99600, 100400) # range(76660, 77460)
    data, norms = load_data('output/data/sun/HOPE_qcGHI_1min_v4.nc', nb_bins=space_size, times=range(ini, ini + time_size, 1), get_norm=True)
    data_mean, data_std = norms

    # plt.imshow(data[0])
    # plt.show()

    mask = torch.zeros_like(data, dtype=torch.bool)
    mask[torch.isnan(data)] = 1
    data[mask] = 0

    sample = torch.stack([data, mask]).unsqueeze(0)

    # arrow("output/data/sun/HOPE_qcGHI_1min_v4.nc", None, times=range(44945 + 4*60 + 25, 45720))
    # arrow("output/data/sun/HOPE_qcGHI_1min_v4.nc", None, times=range(104250, 104840 - 20 - 60))

    # ############################
    # ####### Architecture #######
    # ############################

    CNN = CNN_ST_network(
        in_channels=2,
        hidden=72,
        theta_dim=5,
        spatial_dim=space_dim,
        label="NBE",
    ).to(device)
    CNN.load_state_dict(torch.load(res_path / "networks" / "NBE.pth",map_location=device))
    CNN.eval()

    quantile_CNN_q05 = CNN_ST_network(
        in_channels=2,
        hidden=36,
        theta_dim=5,
        spatial_dim=space_dim,
        label="NBE"
    ).to(device)
    quantile_CNN_q05.load_state_dict(torch.load(res_path / "networks" / "q0.05 - NBE.pth",map_location=device))

    quantile_CNN_q95 = CNN_ST_network(
        in_channels=2,
        hidden=36,
        theta_dim=5,
        spatial_dim=space_dim,
        label="NBE"
    ).to(device)
    quantile_CNN_q95.load_state_dict(torch.load(res_path / "networks" / "q0.95 - NBE.pth",map_location=device))

    tharte = CNN(sample)[0]
    print(tharte)

    # ########################
    # ####### Fenetres #######
    # ########################

    sliding_window(
        output_path,
        104000,104840,
        CNN,
        quantile_CNN_q05,
        quantile_CNN_q95,
        space_size = 50,
    )


    target = 104120
    tharte, q05, q95 = get_gaussian_weighted_estimate(
        p_target=target,
        CNN = CNN,
        quantile_CNN_low = quantile_CNN_q05,
        quantile_CNN_high = quantile_CNN_q95,
        wide=20,
        n_lags=8,
        sigma=2.5,
    )
    for name, m, low, high in zip(theta_names, tharte, q05, q95):
        print(f"{name:8s} : {m:.4f}  [-{low:.4f}, +{high:.4f}]")




    ####################
    ####### Krig #######
    ####################


    # tharte = torch.tensor([-0.6936,  0.5245,  0.4982,  0.2751,  0.0176]) # [-0.7661,  0.7283,  0.4552,  0.1381,  0.0655]
    # tharte = res_wide[wide][-11]

    # print(tharte)
    #
    # time_size = 20
    # space_dim = 2
    # space_size = 50
    #
    # data, norms = load_data('output/data/sun/HOPE_qcGHI_1min_v4.nc', nb_bins=space_size,
    #                         times=range(ini, ini + time_size, 1), get_norm=True)
    # data_mean, data_std = norms
    #
    # # plt.imshow(data[0])
    # # plt.show()
    #
    # mask = torch.zeros_like(data, dtype=torch.bool)
    # mask[torch.isnan(data)] = 1
    # data[mask] = 0
    #
    # sample = torch.stack([data, mask]).unsqueeze(0)
    #
    # tharte = tharte.detach().numpy()
    #
    # sample = sample[0]
    #
    # y = sample[0].reshape(-1)  # order="C"
    # mask_obs = sample[1].to(torch.bool).reshape(-1)
    #
    # t = torch.linspace(0, 1, time_size + 1)[:-1]
    # s = torch.linspace(0, 1, space_size + 1)[:-1]
    #
    # if space_dim == 1:
    #     grid_t, grid_s = torch.meshgrid(t, s, indexing="ij")
    #     coords = torch.stack([grid_s, grid_t], dim=-1).reshape(-1, 2)
    # else:
    #     grid_t, grid_s1, grid_s2 = torch.meshgrid(t, s, s, indexing="ij")
    #     coords = torch.stack([grid_s2, grid_s1, grid_t], dim=-1).reshape(-1, 3)
    #
    # y_obs = y[~mask_obs].detach().numpy()
    # coords_obs = coords[~mask_obs].detach().numpy()
    # coords_miss = coords.detach().numpy() # coords[mask_obs].detach().numpy()
    #
    # coords_miss_future = coords_miss.copy()
    # coords_miss_future[:,space_dim] += 1
    # coords_miss = np.concatenate([coords_miss, coords_miss_future])   # !!!
    # # coords_miss = np.concatenate([coords_miss[mask_obs], coords_miss_future])
    #
    # krig = model.kriging(coords_obs, coords_miss, y_obs, space_dim, tharte)
    #
    # # y_full = torch.cat([y.clone().float(), torch.zeros_like(y)])
    # # y_full[torch.cat([mask_obs, torch.ones_like(mask_obs)])] = torch.tensor(krig).float()
    # y_full = torch.tensor(krig).float()  # !!!
    #
    # print("fin krigeage")
    #
    # print(np.nanmin(y_obs), np.nanmax(y_obs))
    # print(np.nanmin(y_full), np.nanmax(y_full))
    #
    # ####################
    # ####### Simu #######
    # ####################
    #
    # simu_cond = model.simulate({"adv1": tharte[0],"adv2": tharte[1],"range_time": tharte[2],"range_space": tharte[3]}, time_size, space_dim, space_size, seed=None, mult_time=2)[0][0]
    #
    # y_cond = simu_cond.reshape(-1)  # order="C"
    #
    # t = torch.linspace(0, 1, time_size + 1)[:-1]
    # s = torch.linspace(0, 1, space_size + 1)[:-1]
    #
    # y_obs_cond = y_cond[:y_cond.shape[0]//2][~mask_obs]
    #
    # krig_cond = model.kriging(
    #     coords_obs, coords_miss, y_obs_cond, space_dim, tharte
    # )
    #
    # # y_full_cond = torch.tensor(y_cond).clone().float()
    # # y_full_cond[mask_obs] = torch.tensor(krig_cond).float()
    # y_full_cond = torch.tensor(krig_cond).float()   # !!!
    #
    # print("fin simu cond")
    #
    # # ####################
    # # ####### Plot #######
    # # ####################
    #
    # ### Preparation ###
    #
    # data = data_std * data + data_mean
    # simu_cond = data_std * simu_cond + data_mean
    # y = data_std * y + data_mean
    # y_cond = data_std * y_cond + data_mean
    # y_full = data_std * y_full + data_mean
    # y_full_cond= data_std * y_full_cond + data_mean
    # krig = data_std * krig + data_mean
    # krig_cond = data_std * krig_cond + data_mean
    #
    # data_nan = data.clone()
    # data_nan[mask] = torch.nan
    #
    # data_future, (data_future_mean, data_future_std) = load_data('output/data/sun/HOPE_qcGHI_1min_v4.nc', nb_bins=space_size,
    #                         times=range(ini + time_size, ini + time_size*2, 1), get_norm=True)
    # data_future = data_future_std * data_future + data_future_mean
    # data_nan = torch.concatenate([data_nan, data_future])
    #
    # model.add_noise = True
    #
    # y_full_cond_reshaped = y_full_cond.reshape(time_size * 2, space_size, space_size)
    # y_full_reshaped = y_full.reshape(time_size * 2, space_size, space_size)
    #
    # ### Plot krig ###
    #
    # start_time = datetime.strptime("7:21:00 29 June 2013", "%H:%M:%S %d %B %Y")
    #
    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 7), layout="constrained")
    # title = fig.suptitle("")
    #
    # # Griser le fond des graphiques pour que les pixels NaN (transparents) apparaissent en gris
    # axs[2].set_facecolor("silver")  # "gray", "silver" ou "lightgray" au choix
    #
    # if torch.is_tensor(y_full_reshaped):
    #     y_full_reshaped = y_full_reshaped.detach().cpu().numpy()
    # if torch.is_tensor(y_full_cond_reshaped):
    #     y_full_cond_reshaped = y_full_cond_reshaped.detach().cpu().numpy()
    # if torch.is_tensor(data_nan):
    #     data_nan = data_nan.detach().cpu().numpy()
    # if torch.is_tensor(simu_cond):
    #     simu_cond = simu_cond.detach().cpu().numpy()
    # if torch.is_tensor(y_full_cond):
    #     y_full_cond = y_full_cond.detach().cpu().numpy()
    # if torch.is_tensor(y_full):
    #     y_full = y_full.detach().cpu().numpy()
    #
    # res_full = simu_cond - y_full_cond_reshaped + y_full_reshaped
    #
    # mae_full = np.sqrt((y_full_reshaped - data_nan)**2) # np.abs(y_full_reshaped - data_nan)
    # mae_max = np.nanmax(mae_full)
    #
    # v_min, v_max = np.nanmin(y_full_reshaped), np.nanmax(y_full_reshaped)
    #
    # ### INITIALISATION DES GRAPHIQUES (Frame 0) ###
    #
    # # 1. Observations
    # im_obs = axs[0].imshow(
    #     np.flip(data_nan[0].T, axis=0),
    #     aspect="equal",
    #     extent=[0, 1, 0, 1],
    #     vmin=v_min,
    #     vmax=v_max,
    #     cmap=CMAP,
    # )
    # axs[0].set_title("Observations")
    #
    # im_krig = axs[1].imshow(
    #     np.flip(y_full_reshaped[0].T, axis=0),
    #     aspect="equal",
    #     extent=[0, 1, 0, 1],
    #     vmin=v_min,
    #     vmax=v_max,
    #     cmap=CMAP,
    # )
    # axs[1].set_title("Kriging")
    #
    # cols = ["#FFFFFF", "#7AA7F1"]
    # mae_cmap = LinearSegmentedColormap.from_list("custom_mae", cols, N=100)
    # mae_cmap.set_bad(color="silver")
    #
    # im_mae = axs[2].imshow(
    #     np.flip(mae_full[0].T, axis=0),
    #     aspect="equal",
    #     extent=[0, 1, 0, 1],
    #     vmin=0,
    #     vmax=mae_max,
    #     cmap=mae_cmap,
    # )
    # axs[2].set_title("Error")
    #
    # cbar_data = fig.colorbar(im_krig, ax=axs[:2], location="bottom", shrink=0.8, pad=0.05)
    # cbar_data.set_label("GHI")
    #
    # cbar_mae = fig.colorbar(im_mae, ax=axs[2], location="bottom", shrink=0.8, pad=0.05)
    # cbar_mae.set_label("Root Mean Squared Error (RMSE)")
    #
    # def update(t):
    #     im_obs.set_data(np.flip(data_nan[t].T, axis=0))
    #     im_krig.set_data(np.flip(y_full_reshaped[t].T, axis=0))
    #
    #     current_mae = np.flip(mae_full[t].T, axis=0)
    #     im_mae.set_data(current_mae)
    #
    #     if t >= time_size:
    #         axs[0].set_title("Observations\n[ FORECASTING PHASE ]", color="firebrick", fontweight="bold")
    #         axs[1].set_title("Kriging\n[ FORECAST ]", color="firebrick", fontweight="bold")
    #     else:
    #         axs[0].set_title("Observations\n[ OBSERVATION PHASE ]", color="black", fontweight="normal")
    #         axs[1].set_title("Kriging\n[ SPATIAL INTERPOLATION ]", color="black", fontweight="normal")
    #
    #     current_time = start_time + timedelta(minutes=t)
    #     title.set_text(
    #         f"Prediction by kriging\n{model.to_title(tharte)}\n\n"
    #         f"GHI {current_time.strftime('%H:%M:%S %d %B %Y')}"
    #     )
    #
    #     mean_err = np.nanmean(current_mae)
    #     std_err = np.nanstd(current_mae)
    #     axs[2].set_title(
    #         f"Error\nMean : {mean_err:.3f} | Std : {std_err:.3f}",
    #         color="black",
    #         fontweight="normal"
    #     )
    #
    #     return [im_obs, im_krig, im_mae, title]
    #
    #
    # anim = animation.FuncAnimation(fig, update, frames=time_size * 2, interval=200)
    #
    # gif_path = output_path / f"kriging_animated_{date}.gif"
    # anim.save(gif_path, writer="pillow")
    #
    # plt.close(fig)
    #
    # print(gif_path)
    #
    # ### Plot Simu Cond ###
    #
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 5), layout="constrained")
    # title = fig.suptitle("")
    #
    # v_min, v_max = np.nanmin(res_full), np.nanmax(res_full)
    #
    # im_obs = axs[0].imshow(
    #     np.flip(data_nan[0].T, axis=0),
    #     aspect="equal",
    #     extent=[0, 1, 0, 1],
    #     vmin=v_min,
    #     vmax=v_max,
    #     cmap=CMAP
    # )
    # axs[0].set_title("Observations\n[ INTERPOLATION ]")
    #
    # # 2. Simulation Conditionnelle
    # im_simu = axs[1].imshow(
    #     np.flip(res_full[0].T, axis=0),
    #     aspect="equal",
    #     extent=[0, 1, 0, 1],
    #     vmin=v_min,
    #     vmax=v_max,
    #     cmap=CMAP
    # )
    # axs[1].set_title("Conditional Simulation\n[ INTERPOLATION ]")
    #
    # ### COLORBAR ###
    #
    # cbar = fig.colorbar(im_simu, ax=axs, location="bottom", shrink=0.7, pad=0.05)
    # cbar.set_label("GHI")
    #
    #
    # ### FONCTION DE MISE A JOUR ###
    #
    # def update(t):
    #     # Mise à jour des images
    #     im_obs.set_data(np.flip(data_nan[t].T, axis=0))
    #     im_simu.set_data(np.flip(res_full[t].T, axis=0))
    #
    #     # Indication de la phase (Interpolation vs Forecasting)
    #     if t >= time_size:
    #         axs[0].set_title("Observations\n[ FORECASTING PHASE ]", color="firebrick", fontweight="bold")
    #         axs[1].set_title("Conditional Simulation\n[ FORECAST ]", color="firebrick", fontweight="bold")
    #     else:
    #         axs[0].set_title("Observations\n[ OBSERVATION PHASE ]", color="black", fontweight="normal")
    #         axs[1].set_title("Conditional Simulation\n[ SPATIAL INTERPOLATION ]", color="black", fontweight="normal")
    #
    #     # Mise à jour du temps global dans le titre
    #     current_time = start_time + timedelta(minutes=t)
    #     title.set_text(
    #         f"Conditional Simulation\n{model.to_title(tharte)}\n\n"
    #         f"GHI {current_time.strftime('%H:%M:%S %d %B %Y')}"
    #     )
    #
    #     return [im_obs, im_simu, title]
    #
    #
    # # Création de l'animation
    # anim = animation.FuncAnimation(fig, update, frames=time_size * 2, interval=200)
    #
    # # Sauvegarde
    # out_dir = output_path / "examples"
    # out_dir.mkdir(parents=True, exist_ok=True)
    #
    # gif_path = output_path / f"simu_cond_animated_{date}.gif"
    # anim.save(gif_path, writer="pillow")
    #
    # plt.close(fig)
    #
    # print(gif_path)
    #
    # data_to_save = {
    #     "data":data,
    #     "simu_cond":simu_cond,
    #     "y":y,
    #     "y_cond":y_cond,
    #     "krig":krig,
    #     "krig_cond":krig_cond,
    #
    # }
    # torch.save(data_to_save, out_dir / "krig_save.pt")
    #
    # mae_mean_over_time = np.nanmean(mae_full, axis=(1, 2))
    # mae_std_over_time = np.nanstd(mae_full, axis=(1, 2))
    #
    # # 2. Création de l'axe temporel (X)
    # times = [start_time + timedelta(minutes=i) for i in range(len(mae_mean_over_time))]
    #
    # # 3. Création de la figure
    # fig, ax = plt.subplots(figsize=(10, 5))
    #
    # # Tracer la moyenne de l'erreur
    # ax.plot(times, mae_mean_over_time, color="blue", linewidth=2, label="Root Mean Squared Error (RMSE)")
    #
    # # Ajouter un ruban (fill_between) pour l'écart type autour de la moyenne
    # ax.fill_between(
    #     times,
    #     np.maximum(0, mae_mean_over_time - mae_std_over_time),  # On empêche d'aller sous 0
    #     mae_mean_over_time + mae_std_over_time,
    #     color="blue",
    #     alpha=0.2,
    #     label="Std Dev"
    # )
    #
    # # 4. Indication de la séparation Interpolation / Forecasting
    # forecast_start_time = times[time_size-1]
    # ax.axvline(x=forecast_start_time, color="firebrick", linestyle="--", linewidth=2, label="Start of Forecasting")
    #
    # # Griser légèrement la zone de forecasting
    # ax.axvspan(forecast_start_time, times[-1], color='firebrick', alpha=0.05)
    #
    # # 5. Mise en forme du graphique
    # ax.set_title("Evolution of Kriging Error Over Time", fontsize=14, pad=15)
    # ax.set_ylabel("Error (GHI)", fontsize=12)
    # ax.set_xlabel("Time", fontsize=12)
    #
    # # Formatage de l'axe X pour afficher joliment les heures/minutes
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.xticks(rotation=45)
    #
    # ax.grid(True, linestyle=":", alpha=0.7)
    # ax.legend(loc="upper left")
    #
    # plt.tight_layout()
    #
    # # Sauvegarde
    # output_path = Path("./output/results/spatiotemporal").resolve() / "paper/3D"
    # output_path.mkdir(parents=True, exist_ok=True)
    # plt.savefig(output_path / "kriging_error_timeseries.pdf", format="pdf", dpi=300)
    #
    # plt.show()
    #
    # # ####################
    # # ####### honte #######
    # # ####################
    # #
    # # data_nan = data.clone()
    # # data_nan[mask] = torch.nan
    # # data_nan = torch.concatenate([data_nan, data_nan * torch.nan])
    # #
    # # y_full_3d = y_full.reshape(time_size*2, space_size, space_size)
    # # y_full_cond_3d = y_full_cond.reshape(time_size*2, space_size, space_size)
    # # simu_cond_tensor = torch.tensor(simu_cond)
    # #
    # # num_times = 5
    # # times_to_plot = np.linspace(0, time_size*2 - 1, num_times).astype(int)
    # #
    # # v_min = min(y_full.min(), y_full_cond.min(), torch.min(data_nan).item())
    # # v_max = max(y_full.max(), y_full_cond.max(), torch.max(data_nan).item())
    # #
    # # fig, axs = plt.subplots(
    # #     nrows=num_times, ncols=3, figsize=(10, 2.5 * num_times), layout="constrained"
    # # )
    # # fig.suptitle(f"Kriging & Conditional Simulation\n{model.to_title(tharte)}", fontsize=16)
    # #
    # # for i, t in enumerate(times_to_plot):
    # #     im_obs = axs[i, 0].imshow(
    # #         data_nan[t].T.flip(0),
    # #         aspect="equal",
    # #         extent=[0, 1, 0, 1],
    # #         vmin=v_min,
    # #         vmax=v_max,
    # #         cmap=CMAP,
    # #     )
    # #
    # #     im_krig = axs[i, 1].imshow(
    # #         y_full_3d[t].T.flip(0),
    # #         aspect="equal",
    # #         extent=[0, 1, 0, 1],
    # #         vmin=v_min,
    # #         vmax=v_max,
    # #         cmap=CMAP,
    # #     )
    # #
    # #     simu_cond_t = simu_cond_tensor[t] - y_full_cond_3d[t] + y_full_3d[t]
    # #     im_simu = axs[i, 2].imshow(
    # #         simu_cond_t.T.flip(0),
    # #         aspect="equal",
    # #         extent=[0, 1, 0, 1],
    # #         vmin=v_min,
    # #         vmax=v_max,
    # #         cmap=CMAP,
    # #     )
    # #
    # #     if i == 0:
    # #         axs[i, 0].set_title("Observation")
    # #         axs[i, 1].set_title("Kriging")
    # #         axs[i, 2].set_title("Simu Cond")
    # #
    # #     axs[i, 0].set_ylabel(f"Time = {t}", fontsize=12)
    # #
    # #     for j in range(3):
    # #         axs[i, j].set_xticks([])
    # #         if j > 0:
    # #             axs[i, j].set_yticks([])
    # #         else:
    # #             axs[i, j].set_yticks([])
    # #
    # # cbar = fig.colorbar(im_obs, ax=axs, location="right", shrink=0.8, pad=0.02)
    # # cbar.set_label("Valeurs")
    # #
    # # plot_path = output_path / f"grid_comparison_{date}.pdf"
    # #
    # # plt.savefig(plot_path, dpi=150, format="pdf")
    # # plt.show()
    # #
    # # print(f"Image sauvegardée: {plot_path}")
