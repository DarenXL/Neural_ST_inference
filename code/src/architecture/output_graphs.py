import numpy as np
import torch
from matplotlib import pyplot as plt

from utils import THETA_COLORS, baseline_mae, format_seconds


def train_graph(trainers, res_path):

    assert all(item.train_loaders_hist[0].dataset.model.param_names == trainers[0].train_loaders_hist[0].dataset.model.param_names for item in trainers)
    theta_names = trainers[0].train_loaders_hist[0].dataset.model.param_names

    baseline, baseline_theta_mean = baseline_mae(
        trainers[0].test_loaders_hist[0].dataset
    )

    fig_box_train, axs = plt.subplots(1, len(theta_names), figsize=(18, 10))

    losses = []
    labels = []
    for j, trainer in enumerate(trainers):
        if not trainer.architecture.nls:
            labels.append(trainer.architecture.label)
            path = (
                trainer.res_path
                / "loss_evolution"
                / f"{trainer.architecture.label}_history.pt"
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

    fig_box_train.savefig(res_path / "conclusion_graphs" / "score_train.pdf", format="pdf", bbox_inches="tight")
    fig_box_train.savefig(res_path / "conclusion_graphs" / "score_train.png", format="png", bbox_inches="tight")

def test_graph(trainers, res_path):

    assert all(item.train_loaders_hist[0].dataset.model.param_names == trainers[0].train_loaders_hist[0].dataset.model.param_names for item in trainers)
    theta_names = trainers[0].train_loaders_hist[0].dataset.model.param_names

    baseline, baseline_theta_mean = baseline_mae(
        trainers[0].test_loaders_hist[0].dataset
    )

    fig_box, axs = plt.subplots(1, len(theta_names), figsize=(18, 10))

    losses = []
    labels = []
    for j, trainer in enumerate(trainers):
        labels.append(trainer.architecture.label)
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"{trainer.architecture.label}_outputs.pt"
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

    plt.suptitle("Test error comparaison")
    plt.tight_layout()
    plt.show()

    fig_box.savefig(res_path / "conclusion_graphs" / "score_test.pdf", format="pdf", bbox_inches="tight")
    fig_box.savefig(res_path / "conclusion_graphs" / "score_test.png", format="png", bbox_inches="tight")


def time_hist(trainers, res_path):

    simu_times, train_times, test_times = [], [], []
    labels = []
    for j, trainer in enumerate(trainers):
        labels.append(trainer.architecture.label)
        simu_times.append(trainer.train_loaders_hist[0].dataset.time_generation)
        path = (
            trainer.res_path
            / "loss_evolution"
            / f"{trainer.architecture.label}_history.pt"
        )
        train_times.append(torch.load(path)["train_time"])
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"{trainer.architecture.label}_outputs.pt"
        )
        test_times.append(torch.load(path)["test_time"])

    x = np.arange(len(labels))
    width = 0.3

    fig_time, ax = plt.subplots(figsize=(8, 5))

    bar0 = ax.bar(x - width, simu_times, width, label='Simulation', color='forestgreen')
    bar1 = ax.bar(x, train_times, width, label='Train', color='royalblue')
    bar2 = ax.bar(x + width, test_times, width, label='Test', color='firebrick')

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

    fig_time.savefig(res_path / "conclusion_graphs" / "times.pdf", format="pdf", bbox_inches="tight")
    fig_time.savefig(res_path / "conclusion_graphs" / "times.png", format="png", bbox_inches="tight")


def target_graph(trainers, res_path, k=0):

    dataset0 = trainers[0].train_loaders_hist[0].dataset
    theta_names = dataset0.model.param_names
    n_params = len(theta_names)

    lower_bound = dataset0.model.lower_bound
    upper_bound = dataset0.model.upper_bound

    baseline, baseline_theta_mean = baseline_mae(dataset0)

    path0 = (
        trainers[0].res_path
        / "complete_outputs"
        / f"{trainers[0].architecture.label}_outputs.pt"
    )
    true_all_theta = torch.load(path0)["true_all_theta"]

    target_true_theta = true_all_theta[k]

    mask = torch.isclose(true_all_theta, target_true_theta, atol=1e-5).all(dim=1)

    preds_dict = {}

    for idx, trainer in enumerate(trainers):
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"{trainer.architecture.label}_outputs.pt"
        )
        all_preds = torch.load(path)["final_all_theta"]

        filtered_preds = all_preds[mask].detach().cpu().numpy()

        color = getattr(trainer, "color", THETA_COLORS[idx]['test'])

        preds_dict[trainer.architecture.label] = {
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

    (res_path / "conclusion_graphs").mkdir(parents=True, exist_ok=True)
    fig_target.savefig(
        res_path / "conclusion_graphs" / f"target_k{k}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig_target.savefig(
        res_path / "conclusion_graphs" / f"target_k{k}.png",
        format="png",
        bbox_inches="tight",
    )
