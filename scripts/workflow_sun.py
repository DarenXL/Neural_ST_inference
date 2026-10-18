import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import gstlearn.document as gdoc
import copy

gdoc.setNoScroll()

import matplotlib.dates as mdates
from neural_st_inference.utils import *
from neural_st_inference.utils_data_sun import *
from neural_st_inference.models.gneiting_adv import GneitingAdvModel
from neural_st_inference.data.datasets import GenericSimulationDataset, ClassifierSimulationDataset
from neural_st_inference.architecture.train_test import *
from neural_st_inference.mle.mle_test import *
from neural_st_inference.architecture.neural_networks import CNN_ST_network, print_parameters_per_layer
from neural_st_inference.output_graphs import *
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

# Path to network

network_path = Path("./output/results/spatiotemporal").resolve() / "2026-10-09_19-07-30_3D"

# Path to results saving file

output_path = create_res_path("output/results/spatiotemporal")

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Neural Network train options

batch_size = 50
nb_epochs = 500
freq_epochs_test = 10

# Data options

time_size = 20
space_dim = 2
space_size = 50

theta_names = [r"$v_1$", r"$v_2$", r"$\rho_t$", r"$\rho_s$", r"$\sigma$", r"$\tau$"]

if __name__ == "__main__":

    ##############################
    ##############################
    ############ Data ############
    ##############################
    ##############################

    ####################
    ####### Load #######
    ####################

    model = GneitingAdvModel(space_dim=space_dim)
    ini = 104120
    data, norms = load_data('output/data/sun/HOPE_qcGHI_1min_v4.nc', nb_bins=space_size, times=range(ini, ini + time_size, 1), get_norm=True)
    data_mean, data_std = norms

    mask = torch.zeros_like(data, dtype=torch.bool)
    mask[torch.isnan(data)] = 1
    data[mask] = 0

    sample = torch.stack([data, mask]).unsqueeze(0)

    # ############################
    # ####### Architecture #######
    # ############################

    CNN = CNN_ST_network(
        in_channels=2,
        hidden=72,
        theta_dim=6,
        spatial_dim=space_dim,
        label="NBE",
    ).to(device)
    CNN.load_state_dict(torch.load(network_path / "networks" / "NBE.pth", map_location=device))
    CNN.eval()

    quantile_CNN_q05 = CNN_ST_network(
        in_channels=2,
        hidden=36,
        theta_dim=6,
        spatial_dim=space_dim,
        label="NBE"
    ).to(device)
    quantile_CNN_q05.load_state_dict(torch.load(network_path / "networks" / "q0.05 - NBE.pth", map_location=device))

    quantile_CNN_q95 = CNN_ST_network(
        in_channels=2,
        hidden=36,
        theta_dim=6,
        spatial_dim=space_dim,
        label="NBE"
    ).to(device)
    quantile_CNN_q95.load_state_dict(torch.load(network_path / "networks" / "q0.95 - NBE.pth", map_location=device))

    # ###############################
    # ####### Sliding windows #######
    # ###############################

    sliding_window(
        output_path,
        104000,104840,
        CNN,
        quantile_CNN_q05,
        quantile_CNN_q95,
        space_size = 50,
    )

    target = 104120
    theta_est, q05, q95 = get_gaussian_weighted_estimate(
        p_target=target,
        CNN = CNN,
        quantile_CNN_low = quantile_CNN_q05,
        quantile_CNN_high = quantile_CNN_q95,
        wide=20,
        n_lags=8,
        sigma_window=2.5,
    )
    for name, m, low, high in zip(theta_names, theta_est, q05, q95):
        print(f"{name:8s} : {m:.4f}  [-{low:.4f}, +{high:.4f}]")

    ####################
    ####### Krig #######
    ####################

    print(theta_est)

    time_size = 20
    space_dim = 2
    space_size = 50

    # load data as grid

    data, norms = load_data('output/data/sun/HOPE_qcGHI_1min_v4.nc', nb_bins=space_size,
                            times=range(ini, ini + time_size, 1), get_norm=True)
    data_mean, data_std = norms

    # mask data

    mask = torch.zeros_like(data, dtype=torch.bool)
    mask[torch.isnan(data)] = 1
    data[mask] = 0

    # Reshape

    y = data.reshape(-1)  # order="C"
    mask_obs = mask.to(torch.bool).reshape(-1)

    # Coords creation

    t = torch.linspace(0, 1, time_size + 1)[:-1]
    s = torch.linspace(0, 1, space_size + 1)[:-1]

    if space_dim == 1:
        grid_t, grid_s = torch.meshgrid(t, s, indexing="ij")
        coords = torch.stack([grid_s, grid_t], dim=-1).reshape(-1, 2)
    else:
        grid_t, grid_s1, grid_s2 = torch.meshgrid(t, s, s, indexing="ij")
        coords = torch.stack([grid_s2, grid_s1, grid_t], dim=-1).reshape(-1, 3)

    # Observed locations and missing locations

    y_obs = y[~mask_obs].detach().numpy()
    coords_obs = coords[~mask_obs].detach().numpy()
    coords_miss = coords.detach().numpy()

    # Add future (10min) to missing locations

    coords_miss_future = coords_miss.copy()[:coords_miss.shape[0]//2]
    coords_miss_future[:,space_dim] += 1
    coords_miss = np.concatenate([coords_miss, coords_miss_future])

    # Kriging from real observations

    print("KRIGING...")

    theta_dict = dict(zip(["adv1", "adv2", "range_time", "range_space"], theta_est[:-2]))
    krig = model.kriging(coords_obs, coords_miss, y_obs, space_dim, theta_dict, dev=theta_est[-2], noise=theta_est[-1])
    y_full = torch.tensor(krig).float()

    print("END KRIGING")

    ####################
    ####### Simu #######
    ####################

    # Unconditional simulation

    simu_cond = model.simulate({"adv1": theta_est[0], "adv2": theta_est[1], "range_time": theta_est[2], "range_space": theta_est[3]}, time_size, space_dim, space_size, seed=None, mult_time=1.5)[0][0] * theta_est[4]
    y_cond = simu_cond.reshape(-1)  # order="C"

    # Extract at observed locations

    y_obs_cond = y_cond[:2*y_cond.shape[0]//3][~mask_obs]

    # Kriging on unconditional simulation

    print("KRIGING...")

    krig_cond = model.kriging(
        coords_obs, coords_miss, y_obs_cond, space_dim, theta_dict, dev=theta_est[-2], noise=None
    )
    y_full_cond = torch.tensor(krig_cond).float()

    print("END KRIGING")

    ####################
    ####### Plot #######
    ####################

    # Rescale to GHI values

    data = data_std * data + data_mean
    simu_cond = data_std * simu_cond + data_mean
    y = data_std * y + data_mean
    y_cond = data_std * y_cond + data_mean
    y_full = data_std * y_full + data_mean
    y_full_cond= data_std * y_full_cond + data_mean
    krig = data_std * krig + data_mean
    krig_cond = data_std * krig_cond + data_mean

    # Load real future

    data_nan = data.clone()
    data_nan[mask] = torch.nan

    data_future, (data_future_mean, data_future_std) = load_data('output/data/sun/HOPE_qcGHI_1min_v4.nc', nb_bins=space_size,
                            times=range(ini + time_size, ini + 3*time_size//2, 1), get_norm=True)
    data_future = data_future_std * data_future + data_future_mean
    data_nan = torch.concatenate([data_nan, data_future])

    y_full_cond_reshaped = y_full_cond.reshape(int(time_size * 1.5), space_size, space_size)
    y_full_reshaped = y_full.reshape(int(time_size * 1.5), space_size, space_size)

    if torch.is_tensor(y_full_reshaped):
        y_full_reshaped = y_full_reshaped.detach().cpu().numpy()
    if torch.is_tensor(y_full_cond_reshaped):
        y_full_cond_reshaped = y_full_cond_reshaped.detach().cpu().numpy()
    if torch.is_tensor(data_nan):
        data_nan = data_nan.detach().cpu().numpy()
    if torch.is_tensor(simu_cond):
        simu_cond = simu_cond.detach().cpu().numpy()
    if torch.is_tensor(y_full_cond):
        y_full_cond = y_full_cond.detach().cpu().numpy()
    if torch.is_tensor(y_full):
        y_full = y_full.detach().cpu().numpy()

    ### Plot krig ###

    start_time = datetime.strptime("7:21:00 29 June 2013", "%H:%M:%S %d %B %Y")

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 7), layout="constrained")
    title = fig.suptitle("")

    axs[2].set_facecolor("silver")

    res_full = simu_cond - y_full_cond_reshaped + y_full_reshaped

    mae_full = np.sqrt((y_full_reshaped - data_nan)**2) # np.abs(y_full_reshaped - data_nan)
    mae_max = np.nanmax(mae_full)

    v_min, v_max = np.nanmin(y_full_reshaped), np.nanmax(y_full_reshaped)

    im_obs = axs[0].imshow(
        np.flip(data_nan[0].T, axis=0),
        aspect="equal",
        extent=[0, 1, 0, 1],
        vmin=v_min,
        vmax=v_max,
        cmap=CMAP,
    )
    axs[0].set_title("Observations")

    im_krig = axs[1].imshow(
        np.flip(y_full_reshaped[0].T, axis=0),
        aspect="equal",
        extent=[0, 1, 0, 1],
        vmin=v_min,
        vmax=v_max,
        cmap=CMAP,
    )
    axs[1].set_title("Kriging")

    cols = ["#FFFFFF", "#7AA7F1"]
    mae_cmap = LinearSegmentedColormap.from_list("custom_mae", cols, N=100)
    mae_cmap.set_bad(color="silver")

    im_mae = axs[2].imshow(
        np.flip(mae_full[0].T, axis=0),
        aspect="equal",
        extent=[0, 1, 0, 1],
        vmin=0,
        vmax=mae_max,
        cmap=mae_cmap,
    )
    axs[2].set_title("Error")

    cbar_data = fig.colorbar(im_krig, ax=axs[:2], location="bottom", shrink=0.8, pad=0.05)
    cbar_data.set_label("GHI")

    cbar_mae = fig.colorbar(im_mae, ax=axs[2], location="bottom", shrink=0.8, pad=0.05)
    cbar_mae.set_label("Root Mean Squared Error (RMSE)")

    def update(t):
        im_obs.set_data(np.flip(data_nan[t].T, axis=0))
        im_krig.set_data(np.flip(y_full_reshaped[t].T, axis=0))

        current_mae = np.flip(mae_full[t].T, axis=0)
        im_mae.set_data(current_mae)

        if t >= time_size:
            axs[0].set_title("Observations\n[ FORECASTING PHASE ]", color="firebrick", fontweight="bold")
            axs[1].set_title("Kriging\n[ FORECAST ]", color="firebrick", fontweight="bold")
        else:
            axs[0].set_title("Observations\n[ OBSERVATION PHASE ]", color="black", fontweight="normal")
            axs[1].set_title("Kriging\n[ SPATIAL INTERPOLATION ]", color="black", fontweight="normal")

        current_time = start_time + timedelta(minutes=t)
        title.set_text(
            f"Prediction by kriging\n{model.to_title(theta_est) + rf" | $\sigma$: {theta_est[-2]:.2f}" + rf" | $\tau$: {theta_est[-1]:.2f}"}\n\n"
            f"GHI {current_time.strftime('%H:%M:%S %d %B %Y')}"
        )

        mean_err = np.nanmean(current_mae)
        std_err = np.nanstd(current_mae)
        axs[2].set_title(
            f"Error\nMean : {mean_err:.3f} | Std : {std_err:.3f}",
            color="black",
            fontweight="normal"
        )

        return [im_obs, im_krig, im_mae, title]

    anim = animation.FuncAnimation(fig, update, frames=3*time_size // 2, interval=200)

    gif_path = output_path / f"kriging_animated_{date}.gif"
    anim.save(gif_path, writer="pillow")

    plt.close(fig)

    print(gif_path)

    ### Plot Simu Cond ###

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 7), layout="constrained")
    title = fig.suptitle("")

    v_min, v_max = np.nanmin(res_full), np.nanmax(res_full)

    im_obs = axs[0].imshow(
        np.flip(data_nan[0].T, axis=0),
        aspect="equal",
        extent=[0, 1, 0, 1],
        vmin=v_min,
        vmax=v_max,
        cmap=CMAP
    )
    axs[0].set_title("Observations\n[ INTERPOLATION ]")

    im_simu = axs[1].imshow(
        np.flip(res_full[0].T, axis=0),
        aspect="equal",
        extent=[0, 1, 0, 1],
        vmin=v_min,
        vmax=v_max,
        cmap=CMAP
    )
    axs[1].set_title("Conditional Simulation\n[ INTERPOLATION ]")

    cbar = fig.colorbar(im_simu, ax=axs, location="bottom", shrink=0.7, pad=0.05)
    cbar.set_label("GHI")

    def update(t):
        im_obs.set_data(np.flip(data_nan[t].T, axis=0))
        im_simu.set_data(np.flip(res_full[t].T, axis=0))

        if t >= time_size:
            axs[0].set_title("Observations\n[ FORECASTING PHASE ]", color="firebrick", fontweight="bold")
            axs[1].set_title("Conditional Simulation\n[ FORECAST ]", color="firebrick", fontweight="bold")
        else:
            axs[0].set_title("Observations\n[ OBSERVATION PHASE ]", color="black", fontweight="normal")
            axs[1].set_title("Conditional Simulation\n[ SPATIAL INTERPOLATION ]", color="black", fontweight="normal")

        current_time = start_time + timedelta(minutes=t)
        title.set_text(
            f"Conditional Simulation\n{model.to_title(theta_est) + rf" | $\sigma$: {theta_est[-2]:.2f}" + rf" | $\tau$: {theta_est[-1]:.2f}"}\n\n"
            f"GHI {current_time.strftime('%H:%M:%S %d %B %Y')}"
        )

        return [im_obs, im_simu, title]

    anim = animation.FuncAnimation(fig, update, frames=3*time_size // 2, interval=200)

    out_dir = output_path / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    gif_path = output_path / f"simu_cond_animated_{date}.gif"
    anim.save(gif_path, writer="pillow")

    plt.close(fig)

    print(gif_path)

    data_to_save = {
        "data":data,
        "simu_cond":simu_cond,
        "y":y,
        "y_cond":y_cond,
        "krig":krig,
        "krig_cond":krig_cond,

    }
    torch.save(data_to_save, out_dir / "krig_save.pt")

    ### Prediction Error ###

    mae_mean_over_time = np.nanmean(mae_full, axis=(1, 2))
    mae_std_over_time = np.nanstd(mae_full, axis=(1, 2))

    times = [start_time + timedelta(minutes=i) for i in range(len(mae_mean_over_time))]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(times, mae_mean_over_time, color="blue", linewidth=2, label="Root Mean Squared Error (RMSE)")

    ax.fill_between(
        times,
        np.maximum(0, mae_mean_over_time - mae_std_over_time),
        mae_mean_over_time + mae_std_over_time,
        color="blue",
        alpha=0.2,
        label="Std Dev"
    )

    forecast_start_time = times[time_size-1]
    ax.axvline(x=forecast_start_time, color="firebrick", linestyle="--", linewidth=2, label="Start of Forecasting")

    ax.axvspan(forecast_start_time, times[-1], color='firebrick', alpha=0.05)

    ax.set_title("Evolution of Kriging Error Over Time", fontsize=14, pad=15)
    ax.set_ylabel("Error (GHI)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper left")

    plt.tight_layout()

    output_path = Path("./output/results/spatiotemporal").resolve() / "paper/3D"
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "kriging_error_timeseries.pdf", format="pdf", dpi=300)

    plt.show()
