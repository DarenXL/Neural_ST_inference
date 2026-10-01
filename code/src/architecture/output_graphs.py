import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from utils import (
    THETA_COLORS,
    CMAP,
    baseline_mae,
    format_seconds,
)
from architecture.train_test import Trainer


def list_trainers_name(trainers: list[Trainer], num_dataset: list):
    """returns a string containing for each trainer of trainers: its network's label and its dataset it train/test on"""
    return " - ".join(
        [
            trainer.estimation_label + f" [{n}]"
            for n, trainer in zip(num_dataset, trainers)
        ]
    )


def train_graph(
    trainers: list[Trainer], res_path: Path, num_dataset: int | list[int] = 0
):
    """
    boxplot with train scores
    trainers : A list of Trainer
    res_path : path to save results
    num_dataset : used to choose which training dataset the score should be shown on
        if an int => same index is used for every Trainer
        if a list[int] : should be the same size as trainers
    """
    if isinstance(num_dataset, int):
        num_dataset = [num_dataset] * len(trainers)

    # assert all(item.train_dataset_hist[0].model.param_names == trainers[0].train_dataset_hist[0].model.param_names for item in trainers)
    theta_names = trainers[0].train_dataset_hist[num_dataset[0]].model.param_names

    baseline, baseline_theta_mean = baseline_mae(trainers[0].test_dataset_hist[0])

    fig_box_train, axs = plt.subplots(1, len(theta_names), figsize=(18, 7))

    losses = []
    labels = []
    for j, trainer in enumerate(trainers):
        if not trainer.nls:
            labels.append(trainer.estimation_label)
            path = (
                trainer.res_path
                / "loss_evolution"
                / f"history - {trainer.estimation_label} [{num_dataset[j]}].pt"
            )
            loss = torch.load(path)["final_test_losses"]
            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu().numpy()
            losses.append(loss)

    if len(losses) == 0:
        return

    for i, name in enumerate(theta_names):

        axs[i].boxplot(
            [loss[:, i] for loss in losses],
            tick_labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor=THETA_COLORS[i]["train"], alpha=0.6),
            medianprops=dict(linestyle="-", linewidth=2, color="black"),
        )
        axs[i].tick_params(axis="x", labelrotation=45)
        axs[i].set_title("MAE – " + name)
        axs[i].set_ylabel("MAE")

        axs[i].axhline(
            baseline[i],
            color=THETA_COLORS[i]["baseline"],
            ls="--",
            label="baseline " + name,
        )
        # axs[i].legend()

    plt.suptitle("Train error comparaison")
    plt.tight_layout()
    plt.show()

    fig_box_train.savefig(
        res_path
        / "conclusion_graphs"
        / f"score_train - {list_trainers_name(trainers, num_dataset)}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig_box_train.savefig(
        res_path
        / "conclusion_graphs"
        / f"score_train - {list_trainers_name(trainers, num_dataset)}.png",
        format="png",
        bbox_inches="tight",
    )


def test_graph(
    trainers: list[Trainer], res_path: Path, num_dataset: int | list[int] = 0
):
    """
    boxplot with test scores
    trainers : A list of Trainer
    res_path : path to save results
    num_dataset : used to choose which testing dataset the score should be shown on
        if an int => same index is used for every Trainer
        if a list[int] : should be the same size as trainers
    """

    if isinstance(num_dataset, int):
        num_dataset = [num_dataset] * len(trainers)

    # assert all(item.test_dataset_hist[0].model.param_names == trainers[0].test_dataset_hist[0].model.param_names for item in trainers)
    theta_names = (
        trainers[0].test_dataset_hist[0].model.get_param_names(alt_format=True)
    )

    baseline, baseline_theta_mean = baseline_mae(
        trainers[0].test_dataset_hist[num_dataset[0]]
    )

    fig_box, axs = plt.subplots(1, len(theta_names), figsize=(12, 5))

    losses = []
    labels = []
    for j, trainer in enumerate(trainers):
        labels.append(trainer.estimation_label)
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs - {trainer.estimation_label} [{num_dataset[j]}].pt"
        )
        loss = torch.load(path)["final_test_losses"]
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        losses.append(loss)

    for i, name in enumerate(theta_names):

        axs[i].boxplot(
            [loss[:, i] for loss in losses],
            tick_labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor=THETA_COLORS[i]["test"], alpha=0.6),
            medianprops=dict(linestyle="-", linewidth=2, color="black"),
        )
        axs[i].tick_params(axis="x", labelrotation=45)
        axs[i].set_title("MAE – " + name)
        axs[i].set_ylabel("MAE" if i == 0 else "")

        axs[i].axhline(
            baseline[i],
            color=THETA_COLORS[i]["baseline"],
            ls="--",
            label="baseline " + name,
        )
        if i == len(axs) - 1:
            axs[i].legend()

    fig_box.suptitle(
        f"Test error"
    )  #  {trainers[0].test_dataset_hist[0].missing_ratio*100}%") # - ~80% missing values
    fig_box.tight_layout()
    plt.show()

    fig_box.savefig(
        res_path
        / "conclusion_graphs"
        / f"score_test - {list_trainers_name(trainers, num_dataset)}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig_box.savefig(
        res_path
        / "conclusion_graphs"
        / f"score_test - {list_trainers_name(trainers, num_dataset)}.png",
        format="png",
        bbox_inches="tight",
    )


def test_alt_graph(trainers: list[Trainer], res_path: Path):
    """
    For every Trainer of trainers, show a test score boxplot on every test datasets of the Trainer
    trainers : A list of Trainer
    res_path : path to save results
    """

    for trainer in trainers:
        _test_alt_graph(trainer, res_path)


def _test_alt_graph(trainer: Trainer, res_path: Path):
    """
    Show a test score boxplot on every test datasets of the Trainer
    """

    theta_names = trainer.test_dataset_hist[0].model.get_param_names(alt_format=True)

    baseline, baseline_theta_mean = baseline_mae(trainer.test_dataset_hist[0])

    fig_box, axs = plt.subplots(1, len(theta_names), figsize=(13, 5))

    losses = []
    labels = ["~92.0%\nmissing\nratio", "~97.6%\nmissing\nratio", "~99.4%\nmissing\nratio"]
    missing_ratios = []
    for j, dataset in enumerate(trainer.test_dataset_hist):
        # labels.append(f"missing\nratio\n{dataset.missing_ratio}")
        missing_ratios.append(dataset.missing_ratio)
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs - {trainer.estimation_label} [{j}].pt"
        )
        loss = torch.load(path)["final_test_losses"]
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        losses.append(loss)

    for i, name in enumerate(theta_names):

        axs[i].boxplot(
            [loss[:, i] for loss in losses],
            tick_labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor=THETA_COLORS[i]["test"], alpha=0.6),
            medianprops=dict(linestyle="-", linewidth=2, color="black"),
        )
        axs[i].tick_params(axis="x")  # , labelrotation=45)
        axs[i].set_title("MAE – " + name)
        axs[i].set_ylabel("MAE" if i == 0 else "")

        axs[i].axhline(
            baseline[i],
            color=THETA_COLORS[i]["baseline"],
            ls="--",
            label="baseline",  # + name
        )
        if i == len(axs) - 1:
            axs[i].legend()

    fig_box.suptitle(f"Test error comparaison - {trainer.estimation_label}")
    fig_box.tight_layout()
    plt.show()

    fig_box.savefig(
        res_path
        / "conclusion_graphs"
        / f"score_test_alt - {trainer.estimation_label}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig_box.savefig(
        res_path
        / "conclusion_graphs"
        / f"score_test_alt - {trainer.estimation_label}.png",
        format="png",
        bbox_inches="tight",
    )


def time_hist(
    trainers: list[Trainer],
    res_path: Path,
    num_dataset_train: int | list[int] = 0,
    num_dataset_test: int | list[int] = 0,
):
    # todo : séparer num_dataset pour train/test ou faire une somme
    """
    boxplot with simulation, train and test times
    trainers : A list of Trainer
    res_path : path to save results
    num_dataset : used to choose which datasets the times should be shown on
        if an int => same index is used for every Trainer
        if a list[int] : should be the same size as trainers
    """

    if isinstance(num_dataset_train, int):
        num_dataset_train = [num_dataset_train] * len(trainers)

    if isinstance(num_dataset_test, int):
        num_dataset_test = [num_dataset_test] * len(trainers)

    simu_times, train_times, test_times = [], [], []
    labels = []
    for j, trainer in enumerate(trainers):
        labels.append(trainer.estimation_label)
        simu_times.append(trainer.simu_time)
        # train_times.append(trainer.train_time)
        # test_times.append(trainer.test_time)

        # trainer.train_dataset_hist[0].time_generation
        path = (
            trainer.res_path
            / "loss_evolution"
            / f"history - {trainer.estimation_label} [{num_dataset_train[j]}].pt"
        )
        train_times.append(torch.load(path)["train_time"])
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs - {trainer.estimation_label} [{num_dataset_test[j]}].pt"
        )
        test_times.append(torch.load(path)["test_time"])

    x = np.arange(len(labels))
    width = 0.3

    fig_time, ax = plt.subplots(figsize=(8, 5))
    # cols = ["#7AA7F1", "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    bar0 = ax.bar(x - width, simu_times, width, label="Simulation", color="#7AA7F1")
    bar1 = ax.bar(x, train_times, width, label="Train", color="#785EF0")
    bar2 = ax.bar(x + width, test_times, width, label="Test", color="#DC267F")

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

    fig_time.savefig(
        res_path
        / "conclusion_graphs"
        / f"times - {list_trainers_name(trainers, zip(num_dataset_train,num_dataset_test))}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig_time.savefig(
        res_path
        / "conclusion_graphs"
        / f"times - {list_trainers_name(trainers, zip(num_dataset_train,num_dataset_test))}.png",
        format="png",
        bbox_inches="tight",
    )


def target_graph(
    trainers: list[Trainer],
    res_path: Path,
    k: int = 0,
    num_dataset: int | list[int] = 0,
):
    """
    Several scatter plots showing the true values and estimates of each trainer's networks for each deepset for each pair of parameters.
    trainers : A list of Trainer
    res_path : path to save results
    k : index of theta
    num_dataset : used to choose which testing dataset should be shown on
        if an int => same index is used for every Trainer
        if a list[int] : should be the same size as trainers
    """

    if isinstance(num_dataset, int):
        num_dataset = [num_dataset] * len(trainers)

    dataset0 = trainers[0].test_dataset_hist[num_dataset[0]]
    theta_names = dataset0.model.param_names
    n_params = len(theta_names)

    lower_bound = dataset0.model.lower_bound
    upper_bound = dataset0.model.upper_bound

    baseline, baseline_theta_mean = baseline_mae(dataset0)

    path0 = (
        trainers[0].res_path
        / "complete_outputs"
        / f"outputs - {trainers[0].estimation_label} [{num_dataset[0]}].pt"
    )
    true_all_theta = torch.load(path0)["true_all_theta"]

    target_true_theta = true_all_theta[k]

    # print(true_all_theta, target_true_theta)

    mask = torch.isclose(true_all_theta, target_true_theta, atol=1e-5).all(dim=1)

    # print(mask)

    preds_dict = {}

    for j, trainer in enumerate(trainers):
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs - {trainer.estimation_label} [{num_dataset[j]}].pt"
        )
        all_preds = torch.load(path)["final_all_theta"]

        print(mask.shape, all_preds.shape)

        assert (
            mask.shape[0] == all_preds.shape[0]
        ), "Target graph do not work on : \n - test dataset with merge_test_deepset = True \n - deepset random sizes"

        filtered_preds = all_preds[mask].detach().cpu().numpy()

        color = getattr(trainer, "color", THETA_COLORS[-j]["test"])

        preds_dict[trainer.estimation_label] = {
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

    fig_target.savefig(
        res_path
        / "conclusion_graphs"
        / f"targets - {list_trainers_name(trainers, num_dataset)}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig_target.savefig(
        res_path
        / "conclusion_graphs"
        / f"targets - {list_trainers_name(trainers, num_dataset)}.png",
        format="png",
        bbox_inches="tight",
    )


def visualise(trainers, res_path, nb_examples=2, num_dataset=0, k=None, simu_cond=False):
    """
    Visualisation containing observation, true gaussian field, kriging and MAE of kriging
    trainers : A list of Trainer
    res_path : path to save results
    nb_examples : number of examples (number of rows)
    num_dataset : used to choose which testing dataset should be shown on
        if an int => same index is used for every Trainer
        if a list[int] : should be the same size as trainers
    k : index where the first example start
    """

    with torch.no_grad() :

        if isinstance(num_dataset, int):
            num_dataset = [num_dataset] * len(trainers)

        if k is None :
            k = [i for i in range(nb_examples)]

        for j, trainer in enumerate(trainers):
            if not trainer.nls:
                visualise_output_alt(
                    trainer,
                    res_path,
                    nb_examples,
                    num_dataset[j],
                    k,
                    mask=None,
                    simu_cond=simu_cond,
                )
            else:
                return


def visualise_output(
    trainer, res_path, nb_examples=2, num_dataset=0, k=0, mask=None, simu_cond=False
):
    """Visualisation containing observation, true gaussian field, kriging and MAE of kriging"""
    p = 4
    fig_map, axs = plt.subplots(
        nb_examples, p, figsize=(4 * p, nb_examples * 4), constrained_layout=True
    )
    print(nb_examples)
    fig_map.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    dataset = trainer.test_dataset_hist[num_dataset]
    space_dim = dataset.space_dim

    if space_dim == 1:
        label_x = r"$\it{Space}$"
        label_y = r"$\it{Time}$"
    else:
        label_x = r"$\it{Space 1}$"
        label_y = r"$\it{Space 2}$"

    for i in range(nb_examples):
        full_sample, true_theta, _ = dataset[i + k]
        full_sample, true_theta = full_sample[0], true_theta[0]

        axs[i, 1].imshow(
            full_sample[0] if space_dim == 1 else full_sample[0, 0],
            cmap="viridis",
            extent=[0, 1, 0, 1],
        )
        axs[i, 1].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 1].set_title(
            (r"= Simu with true $\theta$ =" + "\n" if i == 0 else "")
            + dataset.model.to_title(true_theta)
        )

        masked_sample = full_sample.clone()
        if not (mask is None):
            masked_sample[1] = mask[i]
        masked_sample[0][np.array(masked_sample[1], dtype=bool)] = 0

        masked_sample_nan = masked_sample.detach().numpy().copy()
        masked_sample_nan[0][np.array(masked_sample[1], dtype=bool)] = np.nan

        axs[i, 0].imshow(
            masked_sample_nan[0] if space_dim == 1 else masked_sample_nan[0, 0],
            cmap="viridis",
            extent=[0, 1, 0, 1],
        )
        axs[i, 0].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 0].set_ylabel(label_y)
        axs[i, 0].set_title(
            (r"= Data with true $\theta$ =" + "\n" if i == 0 else "")
            + dataset.model.to_title(true_theta)
        )

        theta_pred = (
            trainer.network(masked_sample.unsqueeze(0).float())[0].detach().numpy()
        )

        sample_uncond = dataset.generate_from_theta(theta_pred)[0][
            0
        ]  # dataset.generate_one(np.random.randint(dataset.theta_size))[0][0]
        sample_uncond_masked = sample_uncond.copy()
        sample_uncond_masked[np.array(masked_sample[1], dtype=bool)] = 0

        sample_krig_uncond = dataset.kriging(
            np.stack([sample_uncond_masked, masked_sample[1]]),
            theta_pred,
        )

        sample_krig = dataset.kriging(masked_sample.detach().numpy(), theta_pred)

        if simu_cond:
            sample_krig = sample_krig + sample_uncond - sample_krig_uncond

        axs[i, 2].imshow(
            sample_krig if space_dim == 1 else sample_krig[0],
            cmap="viridis",
            extent=[0, 1, 0, 1],
        )
        axs[i, 2].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 2].set_title(
            (r"= Prediction with inferred $\theta$ =" + "\n" if i == 0 else "")
            + dataset.model.to_title(theta_pred)
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
        axs[i, 3].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 3].set_title(
            (r"= Mean Absolute Error =" + "\n" if i == 0 else "")
            + f"Error : {np.mean(abs(sample_krig - full_sample[0].detach().numpy())):.3f}"
            f"\nNormalized error : {np.mean(abs(sample_krig - full_sample[0].detach().numpy()))/np.std(abs(full_sample[0].detach().numpy())):.3f}"
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
        cbar_data.set_label("")  # 'Value'

    mae_images = [axs[i, 3].get_images()[0] for i in range(nb_examples)]
    if mae_images:
        mae_max = max(np.nanmax(im.get_array()) for im in mae_images)

        cols = ["#FFFFFF", "#7AA7F1"]
        data_cmap = LinearSegmentedColormap.from_list("custom_data", cols, N=100)

        for im in mae_images:
            im.set_clim(0, mae_max)
            im.set_cmap(data_cmap)  # "Blues"

        cbar_mae = fig_map.colorbar(
            mae_images[0], ax=axs[:, 3], location="right", shrink=0.8, pad=0.05
        )
        # cbar_mae.set_label("MAE (Error)")

    fig_map.savefig(
        res_path
        / "conclusion_graphs"
        / f"visualisation - {list_trainers_name([trainer], [num_dataset])}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig_map.savefig(
        res_path
        / "conclusion_graphs"
        / f"visualisation - {list_trainers_name([trainer], [num_dataset])}.png",
        format="png",
        bbox_inches="tight",
    )

    plt.show()


def visualise_output_alt(
    trainer, res_path, nb_examples=2, num_dataset=0, k=None, mask=None, simu_cond=False
):
    """Visualisation containing observation, true gaussian field, kriging and MAE of kriging"""
    p = 5
    fig_map, axs = plt.subplots(
        nb_examples, p, figsize=(4 * p, nb_examples * 4.2), constrained_layout=True, squeeze=False
    )
    fig_map.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    dataset = trainer.test_dataset_hist[num_dataset]
    space_dim = dataset.space_dim

    path = (
        trainer.res_path
        / "complete_outputs"
        / f"outputs - {trainer.estimation_label} [{num_dataset}].pt"
    )
    all_theta_pred = torch.load(path)["final_all_theta"]
    true_all_theta = torch.load(path)["true_all_theta"]

    print(all_theta_pred)
    print(true_all_theta)

    if space_dim == 1:
        label_x = r"Space"
        label_y = r"Time"
    else:
        label_x = r"Space 1"
        label_y = r"Space 2"

    for i in range(nb_examples):
        print(k[i])
        full_sample, true_theta = dataset[k[i]]
        full_sample, true_theta = full_sample[0], true_theta[0]
        print(true_theta)

        axs[i, 1].imshow(
            full_sample[0] if space_dim == 1 else full_sample[0, 0],
            cmap="viridis",
            extent=[0, 1, 0, 1],
        )
        axs[i, 1].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 1].set_title(
            (r"= Simulation with true $\theta$ =" + "\n" if i == 0 else "")
            + dataset.model.to_title(true_theta)
        )


        masked_sample = full_sample.clone()
        mask = masked_sample[1].clone().detach().to(torch.bool)
        masked_sample[0][mask] = 0

        masked_sample_nan = masked_sample.detach() # .numpy().copy()
        masked_sample_nan[0][torch.tensor(masked_sample[1], dtype=torch.bool)] = torch.nan

        axs[i, 0].imshow(
            masked_sample_nan[0] if space_dim == 1 else masked_sample_nan[0, 0],
            cmap="viridis",
            extent=[0, 1, 0, 1],
        )
        axs[i, 0].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 0].set_ylabel(label_y)
        axs[i, 0].set_title(
            (r"= Data with true $\theta$ =" + "\n" if i == 0 else "")
            + dataset.model.to_title(true_theta)
        )


        # theta_pred = true_all_theta[k[i]].detach().cpu().numpy() * (
        #     1 + np.random.normal(0, 0.1, size=true_theta.shape)
        # )  # all_theta_pred[i+k].detach().cpu().numpy()

        theta_pred = all_theta_pred[k[i]].detach().cpu().numpy()
        print(theta_pred)

        # sample_uncond = dataset.generate_one(np.random.randint(dataset.theta_size))[0][0]
        sample_uncond = dataset.generate_from_theta(theta_pred)[0][0]
        sample_uncond_masked = sample_uncond.clone()
        mask = masked_sample[1].clone().detach().to(torch.bool)
        sample_uncond_masked[mask] = 0

        sample_krig_uncond = dataset.kriging_from_sample(
            torch.stack([sample_uncond_masked, masked_sample[1]]),
            theta_pred,
        )

        sample_krig = dataset.kriging_from_sample(
            masked_sample, theta_pred
        )

        axs[i, 2].imshow(
            sample_krig if space_dim == 1 else sample_krig[0], extent=[0, 1, 0, 1]
        )
        axs[i, 2].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 2].set_title(
            (r"= Prediction with kriging =" + "\n" if i == 0 else "")
            + dataset.model.to_title(theta_pred)
        )

        sample_krig = sample_krig + sample_uncond - sample_krig_uncond

        axs[i, 3].imshow(
            sample_uncond if space_dim == 1 else sample_uncond[0], extent=[0, 1, 0, 1]
        )
        axs[i, 3].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 3].set_title(
            (r"= Simulation with $\hat{\theta}$ =" + "\n" if i == 0 else "")
            + dataset.model.to_title(theta_pred)
        )

        axs[i, 4].imshow(
            sample_krig if space_dim == 1 else sample_krig[0], extent=[0, 1, 0, 1]
        )
        axs[i, 4].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 4].set_title(
            (r"= Conditional Simulation with $\hat{\theta}$ =" + "\n" if i == 0 else "")
            + dataset.model.to_title(theta_pred)
        )

    data_images = [
        axs[i, j].get_images()[0] for i in range(nb_examples) for j in range(5)
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
            data_images[0], ax=axs[:, :5], location="right", shrink=1, pad=0.05
        )
        cbar_data.set_label("")  # 'Value'

    fig_map.savefig(
        res_path
        / "conclusion_graphs"
        / f"visualisation_alt - {list_trainers_name([trainer], [num_dataset])}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig_map.savefig(
        res_path
        / "conclusion_graphs"
        / f"visualisation_alt - {list_trainers_name([trainer], [num_dataset])}.png",
        format="png",
        bbox_inches="tight",
    )

    plt.show()
