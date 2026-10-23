import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from neural_st_inference.utils import (
    THETA_COLORS,
    CMAP,
    baseline_mae,
    format_seconds,
)
from neural_st_inference.architecture.train_test import Trainer


def list_trainers_name(trainers: list[Trainer], num_dataset: list):
    """returns a string containing for each trainer of trainers: its network's label and its dataset it train/test on"""
    return " - ".join(
        [
            trainer.estimation_label + f" [{n}]"
            for n, trainer in zip(num_dataset, trainers)
        ]
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

    # assert all(item.test_dataset_hist[0].theta_alt_names == trainers[0].test_dataset_hist[0].theta_alt_names for item in trainers)
    theta_alt_names = trainers[0].test_dataset_hist[0].theta_alt_names

    baseline, baseline_theta_mean = baseline_mae(
        trainers[0].test_dataset_hist[num_dataset[0]]
    )

    fig_box, axs = plt.subplots(1, len(theta_alt_names), figsize=(15, 5))

    losses = []
    labels = []
    for j, trainer in enumerate(trainers):
        labels.append(trainer.estimation_label)
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs_{trainer.estimation_label} [{num_dataset[j]}].pt"
        )
        loss = torch.load(path)["final_test_losses"]
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        losses.append(loss)

    for i, name in enumerate(theta_alt_names):

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

    theta_alt_names = trainer.test_dataset_hist[0].theta_alt_names

    baseline, baseline_theta_mean = baseline_mae(trainer.test_dataset_hist[0])

    fig_box, axs = plt.subplots(1, len(theta_alt_names), figsize=(15, 5))

    losses = []
    labels = (
        []
    )  # ["~92.0%\nmissing\nratio", "~97.6%\nmissing\nratio", "~99.4%\nmissing\nratio"]
    missing_ratios = []
    for j, dataset in enumerate(trainer.test_dataset_hist):
        labels.append(f"missing\nratio\n{dataset.missing_ratio}")
        missing_ratios.append(dataset.missing_ratio)
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs_{trainer.estimation_label} [{j}].pt"
        )
        loss = torch.load(path)["final_test_losses"]
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        losses.append(loss)

    for i, name in enumerate(theta_alt_names):

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
            / "complete_outputs"
            / f"outputs_{trainer.estimation_label} [{num_dataset_train[j]}].pt"
        )
        train_times.append(torch.load(path)["train_time"])
        path = (
            trainer.res_path
            / "complete_outputs"
            / f"outputs_{trainer.estimation_label} [{num_dataset_test[j]}].pt"
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


def visualise(
    trainers, res_path, nb_examples=2, num_dataset=0, k=None, simu_cond=False
):
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

    with torch.no_grad():

        if isinstance(num_dataset, int):
            num_dataset = [num_dataset] * len(trainers)

        if k is None:
            k = [i for i in range(nb_examples)]

        for j, trainer in enumerate(trainers):
            if not trainer.nls:
                visualise_output(
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
    trainer, res_path, nb_examples=2, num_dataset=0, k=None, mask=None, simu_cond=False
):
    """Visualisation containing observation, true gaussian field, kriging and MAE of kriging"""
    p = 5
    fig_map, axs = plt.subplots(
        nb_examples,
        p,
        figsize=(4 * p, nb_examples * 4.2),
        constrained_layout=True,
        squeeze=False,
    )
    fig_map.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    dataset = trainer.test_dataset_hist[num_dataset]
    space_dim = dataset.space_dim

    path = (
        trainer.res_path
        / "complete_outputs"
        / f"outputs_{trainer.estimation_label} [{num_dataset}].pt"
    )
    all_theta_pred = torch.load(path)["final_all_theta"]
    true_all_theta = torch.load(path)["true_all_theta"]

    if space_dim == 1:
        label_x = r"Space"
        label_y = r"Time"
    else:
        label_x = r"Space 1"
        label_y = r"Space 2"

    for i in range(nb_examples):
        full_sample, true_theta = dataset[k[i]]
        full_sample, true_theta = full_sample[0], true_theta[0]

        axs[i, 1].imshow(
            full_sample[0] if space_dim == 1 else full_sample[0, 0],
            cmap="viridis",
            extent=[0, 1, 0, 1],
        )
        axs[i, 1].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 1].set_title(
            (r"= Simulation with true $\theta$ =" + "\n" if i == 0 else "")
            + dataset.to_title(true_theta)
        )

        masked_sample = full_sample.clone()
        mask = masked_sample[1].clone().detach().to(torch.bool)
        masked_sample[0][mask] = 0

        masked_sample_nan = masked_sample.detach()  # .numpy().copy()
        masked_sample_nan[0][torch.tensor(masked_sample[1], dtype=torch.bool)] = (
            torch.nan
        )

        axs[i, 0].imshow(
            masked_sample_nan[0] if space_dim == 1 else masked_sample_nan[0, 0],
            cmap="viridis",
            extent=[0, 1, 0, 1],
        )
        axs[i, 0].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 0].set_ylabel(label_y)
        axs[i, 0].set_title(
            (r"= Data with true $\theta$ =" + "\n" if i == 0 else "")
            + dataset.to_title(true_theta)
        )

        # theta_pred = true_all_theta[k[i]].detach().cpu().numpy() * (
        #     1 + np.random.normal(0, 0.1, size=true_theta.shape)
        # )  # all_theta_pred[i+k].detach().cpu().numpy()

        theta_pred = all_theta_pred[k[i]].detach().cpu().numpy()

        # sample_uncond = dataset.generate_one(np.random.randint(dataset.theta_size))[0][0]
        sample_uncond = dataset.generate_from_theta(theta_pred)[0][0]
        if not (dataset.dev is None):
            sample_uncond *= theta_pred[dataset.sigma_pos]

        sample_uncond_masked = sample_uncond.clone()
        # mask = masked_sample[1].clone().detach().to(torch.bool)
        sample_uncond_masked[mask] = 0

        sample_krig_uncond = dataset.kriging_from_sample(
            torch.stack([sample_uncond_masked, masked_sample[1]]),
            theta_pred,
            keep_noise=False,
        )

        sample_krig = dataset.kriging_from_sample(masked_sample, theta_pred)

        axs[i, 2].imshow(
            sample_krig if space_dim == 1 else sample_krig[0], extent=[0, 1, 0, 1]
        )
        axs[i, 2].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 2].set_title(
            (r"= Prediction with kriging =" + "\n" if i == 0 else "")
            + dataset.to_title(theta_pred)
        )

        sample_krig = sample_krig + sample_uncond - sample_krig_uncond

        axs[i, 3].imshow(
            sample_uncond if space_dim == 1 else sample_uncond[0], extent=[0, 1, 0, 1]
        )
        axs[i, 3].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 3].set_title(
            (r"= Simulation with $\hat{\theta}$ =" + "\n" if i == 0 else "")
            + dataset.to_title(theta_pred)
        )

        axs[i, 4].imshow(
            sample_krig if space_dim == 1 else sample_krig[0], extent=[0, 1, 0, 1]
        )
        axs[i, 4].set_xlabel(label_x if i == (nb_examples - 1) else "")
        axs[i, 4].set_title(
            (r"= Conditional Simulation with $\hat{\theta}$ =" + "\n" if i == 0 else "")
            + dataset.to_title(theta_pred)
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
