import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils import THETA_COLORS, CMAP, baseline_mae, format_seconds


def list_trainers_name(trainers, num_dataset):
    return " - ".join([trainer.network.label + f" [{n}]" for n, trainer in zip(num_dataset,trainers)])

def train_graph(trainers, res_path, num_dataset=0):

    if isinstance(num_dataset, int) :
        num_dataset = [num_dataset]*len(trainers)

    # assert all(item.train_dataset_hist[0].model.param_names == trainers[0].train_dataset_hist[0].model.param_names for item in trainers)
    theta_names = trainers[0].train_dataset_hist[num_dataset[0]].model.param_names

    baseline, baseline_theta_mean = baseline_mae(
        trainers[0].test_dataset_hist[0]
    )

    fig_box_train, axs = plt.subplots(1, len(theta_names), figsize=(18, 7))

    losses = []
    labels = []
    for j, trainer in enumerate(trainers):
        if not trainer.network.nls:
            labels.append(trainer.network.label)
            path = (
                trainer.res_path
                / "loss_evolution"
                / f"history - {trainer.network.label} [{num_dataset[j]}].pt"
            )
            losses.append(torch.load(path)["final_train_losses"])

    for i, name in enumerate(theta_names):

        axs[i].boxplot(
            [loss[:,i] for loss in losses],
            tick_labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor=THETA_COLORS[i]["train"], alpha=0.6),
            medianprops=dict(linestyle="-", linewidth=2, color="black"),
        )
        axs[i].tick_params(axis="x", labelrotation=45)
        axs[i].set_title("MAE – " + name)
        axs[i].set_ylabel("MAE")

        axs[i].axhline(
            baseline[i], color=THETA_COLORS[i]["baseline"], ls="--", label="baseline " + name
        )
        # axs[i].legend()

    plt.suptitle("Train error comparaison")
    plt.tight_layout()
    plt.show()

    fig_box_train.savefig(res_path / "conclusion_graphs" / f"score_train - {list_trainers_name(trainers, num_dataset)}.pdf", format="pdf", bbox_inches="tight")
    fig_box_train.savefig(res_path / "conclusion_graphs" / f"score_train - {list_trainers_name(trainers, num_dataset)}.png", format="png", bbox_inches="tight")

def test_graph(trainers, res_path, num_dataset=0):

    if isinstance(num_dataset, int) :
        num_dataset = [num_dataset]*len(trainers)

    # assert all(item.test_dataset_hist[0].model.param_names == trainers[0].test_dataset_hist[0].model.param_names for item in trainers)
    theta_names = trainers[0].test_dataset_hist[num_dataset[0]].model.param_names

    baseline, baseline_theta_mean = baseline_mae(
        trainers[0].test_dataset_hist[num_dataset[0]]
    )

    fig_box, axs = plt.subplots(1, len(theta_names), figsize=(18, 7))

    losses = []
    labels = []
    for j, trainer in enumerate(trainers):
        labels.append(trainer.network.label)
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs - {trainer.network.label} [{num_dataset[j]}].pt"
        )
        losses.append(torch.load(path)["final_test_losses"])

    for i, name in enumerate(theta_names):

        axs[i].boxplot(
            [loss[:,i] for loss in losses],
            tick_labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor=THETA_COLORS[i]["test"], alpha=0.6),
            medianprops=dict(linestyle="-", linewidth=2, color="black"),
        )
        axs[i].tick_params(axis="x", labelrotation=45)
        axs[i].set_title("MAE – " + name)
        axs[i].set_ylabel("MAE")

        axs[i].axhline(
            baseline[i], color=THETA_COLORS[i]["baseline"], ls="--", label="baseline " + name
        )
        # axs[i].legend()

    fig_box.suptitle(f"Test error comparaison")
    fig_box.tight_layout()
    plt.show()

    fig_box.savefig(res_path / "conclusion_graphs" / f"score_test - {list_trainers_name(trainers, num_dataset)}.pdf", format="pdf", bbox_inches="tight")
    fig_box.savefig(res_path / "conclusion_graphs" / f"score_test - {list_trainers_name(trainers, num_dataset)}.png", format="png", bbox_inches="tight")


def test_alt_graph(trainers, res_path):
    for trainer in trainers :
        _test_alt_graph(trainer, res_path)

def _test_alt_graph(trainer, res_path):

    theta_names = trainer.test_dataset_hist[0].model.get_param_names(alt_format=True)

    baseline, baseline_theta_mean = baseline_mae(
        trainer.test_dataset_hist[0]
    )

    fig_box, axs = plt.subplots(1, len(theta_names), figsize=(13, 5))


    losses = []
    labels = []#["~79%\nmissing\nratio", "~47%\nmissing\nratio", "~16%\nmissing\nratio"]
    missing_ratios = []
    for j, dataset in enumerate(trainer.test_dataset_hist):
        labels.append(f"missing\nratio\n{dataset.missing_ratio}")
        missing_ratios.append(dataset.missing_ratio)
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs - {trainer.network.label} [{j}].pt"
        )
        losses.append(torch.load(path)["final_test_losses"])

    for i, name in enumerate(theta_names):

        axs[i].boxplot(
            [loss[:,i] for loss in losses],
            tick_labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor=THETA_COLORS[i]["test"], alpha=0.6),
            medianprops=dict(linestyle="-", linewidth=2, color="black"),
        )
        axs[i].tick_params(axis="x")# , labelrotation=45)
        axs[i].set_title("MAE – " + name)
        axs[i].set_ylabel("MAE" if i == 0 else "")

        axs[i].axhline(
            baseline[i], color=THETA_COLORS[i]["baseline"], ls="--", label="baseline"  # + name
        )
        if i == len(axs) - 1 :
            axs[i].legend()

    fig_box.suptitle(f"Test error comparaison - {trainer.network.label}")
    fig_box.tight_layout()
    plt.show()

    fig_box.savefig(res_path / "conclusion_graphs" / f"score_test_alt - {trainer.network.label}.pdf", format="pdf", bbox_inches="tight")
    fig_box.savefig(res_path / "conclusion_graphs" / f"score_test_alt - {trainer.network.label}.png", format="png", bbox_inches="tight")

def time_hist(trainers, res_path, num_dataset=0):

    if isinstance(num_dataset, int) :
        num_dataset = [num_dataset]*len(trainers)

    simu_times, train_times, test_times = [], [], []
    labels = []
    for j, trainer in enumerate(trainers):
        labels.append(trainer.network.label)
        simu_times.append(trainer.train_dataset_hist[0].time_generation)
        path = (
            trainer.res_path
            / "loss_evolution"
            / f"history - {trainer.network.label} [{num_dataset[j]}].pt"
        )
        train_times.append(torch.load(path)["train_time"])
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs - {trainer.network.label} [{num_dataset[j]}].pt"
        )
        test_times.append(torch.load(path)["test_time"])

    x = np.arange(len(labels))
    width = 0.3

    fig_time, ax = plt.subplots(figsize=(8, 5))
    # cols = ["#7AA7F1", "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    bar0 = ax.bar(x - width, simu_times, width, label='Simulation', color="#785EF0")
    bar1 = ax.bar(x, train_times, width, label='Train', color="#DC267F")
    bar2 = ax.bar(x + width, test_times, width, label='Test', color="#FE6100")

    for b, t in zip([bar0, bar1, bar2], [simu_times, train_times, test_times]):
        labels_time = [format_seconds(val) for val in t]
        ax.bar_label(b, labels=labels_time, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelrotation=45)

    ax.set_ylabel("Time")
    ax.set_title("Computation time comparison")
    ax.legend()

    ax.margins(y=0.2)

    plt.tight_layout()
    plt.show()

    fig_time.savefig(res_path / "conclusion_graphs" / f"times - {list_trainers_name(trainers, num_dataset)}.pdf", format="pdf", bbox_inches="tight")
    fig_time.savefig(res_path / "conclusion_graphs" / f"times - {list_trainers_name(trainers, num_dataset)}.png", format="png", bbox_inches="tight")


def target_graph(trainers, res_path, k=0, num_dataset=0):

    if isinstance(num_dataset, int) :
        num_dataset = [num_dataset]*len(trainers)

    dataset0 = trainers[0].test_dataset_hist[num_dataset[0]]
    theta_names = dataset0.model.param_names
    n_params = len(theta_names)

    lower_bound = dataset0.model.lower_bound
    upper_bound = dataset0.model.upper_bound

    baseline, baseline_theta_mean = baseline_mae(dataset0)

    path0 = (
        trainers[0].res_path
        / "complete_outputs"
        / f"outputs - {trainers[0].network.label} [{num_dataset[0]}].pt"
    )
    true_all_theta = torch.load(path0)["true_all_theta"]

    target_true_theta = true_all_theta[k]

    mask = torch.isclose(true_all_theta, target_true_theta, atol=1e-5).all(dim=1)

    preds_dict = {}

    for j, trainer in enumerate(trainers):
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs - {trainer.network.label} [{num_dataset[j]}].pt"
        )
        all_preds = torch.load(path)["final_all_theta"]

        filtered_preds = all_preds[mask].detach().cpu().numpy()

        color = getattr(trainer, "color", THETA_COLORS[-j]['test'])

        preds_dict[trainer.network.label] = {
            "pred": filtered_preds,
            "color": color,
        }

    target_true_theta = target_true_theta.detach().cpu().numpy()

    fig_target, axs = plt.subplots(
        n_params - 1, n_params - 1, figsize=(15, 15), constrained_layout=True
    )

    if n_params == 2:
        axs = np.array([[axs]])

    for i in range(1, n_params):
        for j in range(n_params - 1):

            ax = axs[i - 1, j]

            if j >= i:
                ax.axis("off")
                continue

            is_first_plot = i == 1 and j == 0

            ax.scatter(
                target_true_theta[j],
                target_true_theta[i],
                s=500,
                color="black",
                marker="+",
                linewidth=3,
                zorder=10,
                label=r"True $\theta$" if is_first_plot else "",
            )

            ax.scatter(
                baseline_theta_mean[j],
                baseline_theta_mean[i],
                s=250,
                color="gray",
                marker="x",
                linewidth=2,
                alpha=0.7,
                label="Baseline" if is_first_plot else "",
            )

            for label, item in preds_dict.items():
                pred = item["pred"]
                color = item["color"]

                ax.scatter(
                    pred[:, j],
                    pred[:, i],
                    s=200,
                    color=color,
                    marker="o",
                    alpha=0.8,
                    label=label if is_first_plot else "",
                )

            if i == n_params - 1:
                ax.set_xlabel(theta_names[j], fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(theta_names[i], fontsize=12, fontweight="bold")

            ax.grid(True, linestyle="--", alpha=0.3)

            ax.set_xlim([lower_bound[j] - 0.05, upper_bound[j] + 0.05])
            ax.set_ylim([lower_bound[i] - 0.05, upper_bound[i] + 0.05])

    fig_target.suptitle(f"Prediction consistency on sample k={k}", fontsize=16)

    handles, labels_text = axs[0, 0].get_legend_handles_labels()

    if n_params > 2:
        axs[0, 1].legend(
            handles,
            labels_text,
            loc="center left",
            fontsize=12,
            title="Models",
            frameon=True,
        )
    else:
        axs[0, 0].legend()

    plt.show()

    fig_target.savefig(res_path / "conclusion_graphs" / f"targets - {list_trainers_name(trainers, num_dataset)}.pdf", format="pdf", bbox_inches="tight")
    fig_target.savefig(res_path / "conclusion_graphs" / f"targets - {list_trainers_name(trainers, num_dataset)}.png", format="png", bbox_inches="tight")

def visualise(trainers, res_path, nb_examples=3, num_dataset=0, k=0):
    if isinstance(num_dataset, int) :
        num_dataset = [num_dataset]*len(trainers)

    mask = []
    for i in range(nb_examples):
        full_sample, _, _ = trainers[0].test_dataset_hist[num_dataset[0]][i]
        mask.append(full_sample[0][1])

    for j, trainer in enumerate(trainers) :
        if not trainer.network.nls:
            visualise_output(trainer, res_path, nb_examples, num_dataset[j], k, mask=mask)
        else :
            return

def visualise_output(trainer, res_path, nb_examples=3, num_dataset=0, k=0, mask=None):
    p = 4
    fig_map, axs = plt.subplots(
        nb_examples, p, figsize=(4 * p, nb_examples * 3.5), constrained_layout=True
    )
    fig_map.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    dataset = trainer.test_dataset_hist[num_dataset]
    space_dim = dataset.space_dim

    if space_dim == 1 :
        label_x = r"$\it{Space}$"
        label_y = r"$\it{Time}$"
    else:
        label_x = r"$\it{Space 1}$"
        label_y = r"$\it{Space 2}$"

    for i in range(nb_examples):
        full_sample, true_theta, _ = dataset[i+k]
        full_sample, true_theta = full_sample[0], true_theta[0]

        axs[i, 1].imshow(full_sample[0] if space_dim == 1 else full_sample[0,0], cmap="viridis", extent=[0, 1, 0, 1])
        axs[i, 1].set_xlabel(label_x if i == (nb_examples -1) else "")
        axs[i, 1].set_title(
            (r"= Simu with true $\theta$ =" +"\n" if i == 0 else "") + dataset.model.to_title(true_theta,alt_format=True)
        )

        masked_sample = full_sample.clone()
        if not(mask is None):
            masked_sample[1] = mask[i]
        masked_sample[0][np.array(masked_sample[1], dtype=bool)] = 0

        masked_sample_nan = masked_sample.detach().numpy().copy()
        masked_sample_nan[0][np.array(masked_sample[1], dtype=bool)] = np.nan

        axs[i, 0].imshow(masked_sample_nan[0] if space_dim == 1 else masked_sample_nan[0,0], cmap="viridis", extent=[0, 1, 0, 1])
        axs[i, 0].set_xlabel(label_x if i == (nb_examples -1) else "")
        axs[i, 0].set_ylabel(label_y)
        axs[i, 0].set_title(
            (r"= Data with true $\theta$ =" +"\n" if i == 0 else "") + dataset.model.to_title(true_theta,alt_format=True)
        )

        theta_pred = (
            trainer.network(
                masked_sample.unsqueeze(0).float(),
                deepset=torch.tensor([0]),
            )[0]
            .detach()
            .numpy()
        )

        sample_uncond = dataset.generate_one(np.random.randint(dataset.theta_size))[0][0]
        sample_uncond_masked = sample_uncond.copy()
        sample_uncond_masked[np.array(masked_sample[1], dtype=bool)] = 0

        sample_krig_uncond = dataset.kriging(
            np.stack([sample_uncond_masked, masked_sample[1]]),
            theta_pred,
        )

        sample_krig = dataset.kriging(
            masked_sample.detach().numpy(), theta_pred
        )

        sample_pred = sample_krig + sample_uncond - sample_krig_uncond

        axs[i, 2].imshow(
            sample_krig if space_dim == 1 else sample_krig[0], cmap="viridis", extent=[0, 1, 0, 1]
        )
        axs[i, 2].set_xlabel(label_x if i == (nb_examples -1) else "")
        axs[i, 2].set_title(
            (r"= Prediction with inferred $\theta$ =" +"\n" if i == 0 else "") + dataset.model.to_title(theta_pred,alt_format=True)
        )

        mae_data = (
            abs(sample_krig - full_sample[0].detach().numpy())
            if space_dim == 1
            else abs(sample_krig[0] - full_sample[0, 0].detach().numpy())
        )
        axs[i, 3].imshow(
            mae_data,
            cmap="inferno",
            extent=[0, 1, 0, 1],
        )
        axs[i, 3].set_xlabel(label_x if i == (nb_examples -1) else "")
        axs[i, 3].set_title((r"= Mean Absolute Error =" +"\n" if i == 0 else "")
                            +f"Error : {np.mean(abs(sample_krig - full_sample[0].detach().numpy())):.3f}"
                             f"\nNormalized error : {np.mean(abs(sample_krig - full_sample[0].detach().numpy()))
                                                     /np.std(abs(full_sample[0].detach().numpy())):.3f}"
                            )

    data_images = [
        axs[i, j].get_images()[0] for i in range(nb_examples) for j in range(3)
    ]
    if data_images:
        v_min = min(np.nanmin(im.get_array()) for im in data_images)
        v_max = max(np.nanmax(im.get_array()) for im in data_images)

        # cols = ["#7AA7F1", "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
        # data_cmap = LinearSegmentedColormap.from_list("custom_data", cols, N=100)

        for im in data_images:
            im.set_clim(v_min, v_max)
            im.set_cmap(CMAP)

        cbar_data = fig_map.colorbar(
            data_images[0], ax=axs[:, :3], location="right", shrink=0.8, pad=0.05
        )
        cbar_data.set_label("") # 'Value'

    mae_images = [axs[i, 3].get_images()[0] for i in range(nb_examples)]
    if mae_images:
        mae_max = max(np.nanmax(im.get_array()) for im in mae_images)

        cols = ["#FFFFFF", "#7AA7F1"]
        data_cmap = LinearSegmentedColormap.from_list("custom_data", cols, N=100)

        for im in mae_images:
            im.set_clim(0, mae_max)
            im.set_cmap(
                data_cmap # "Blues"
            )

        cbar_mae = fig_map.colorbar(
            mae_images[0], ax=axs[:, 3], location="right", shrink=0.8, pad=0.05
        )
        # cbar_mae.set_label("MAE (Error)")

    fig_map.savefig(res_path / "conclusion_graphs" / f"visualisation - {list_trainers_name([trainer], [num_dataset])}.pdf", format="pdf", bbox_inches="tight")
    fig_map.savefig(res_path / "conclusion_graphs" / f"visualisation - {list_trainers_name([trainer], [num_dataset])}.png", format="png", bbox_inches="tight")

    plt.show()
