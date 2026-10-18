import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import gstlearn.document as gdoc
import copy

gdoc.setNoScroll()

from neural_st_inference.utils import *
from neural_st_inference.output_graphs import *
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        # "font.family": "sherif"
    }
)

#########################
####### Variables #######
#########################

# Neural Network train options

batch_size = 50
nb_epochs = 500
freq_epochs_test = 10

# Data options

time_size = 100
space_dim = 1
space_size = 100
missing_ratios = [0.96]  # 0.94,0.96, 0.98

# Amount of data

train_size = 5
test_size = 2000

# model = "adv"
# inf = "optim"
# if model == "sep":
#     baseline = [0.2359, 0.0255, 0.0625]
#     res_path = (
#         Path("./output/results/spatiotemporal").resolve()
#         / f"2026-08-03_15-23-11_{model}-{inf}"
#     )
# else:
#     baseline = [0.50, 0.025, 0.0625]
#     res_path = (
#         Path("./output/results/spatiotemporal").resolve()
#         / f"2026-08-03_15-23-11_{model}-{inf}"
#     )

model = "adv"
inf = "mcmc"
baseline = [0.50, 0.50, 0.1125, 0.1125, 0.2285, 0.125]
res_path = (
    Path("./output/results/spatiotemporal").resolve()
    / f"2026-10-09_11-31-54_adv-mcmc"
)


char = {"sep": r"$\gamma$", "adv": r"$v$"}
theta_names = [r"$v$",r"$\rho_t$", r"$\rho_s$", r"$\sigma$", r"$\tau$"]

output_path = Path("./output/results/spatiotemporal").resolve() / f"paper" / "3D"


if __name__ == "__main__":

    ##############################
    ##############################
    ############ Test ############
    ##############################
    ##############################

    ######################
    ####### Median #######
    ######################

    # print(res_path)
    #
    # fig_box, axs = plt.subplots(1, len(theta_names), figsize=(12, 5))
    #
    # losses = []
    # labels = ["NBE", "NRE"]
    # # labels = ["NRE\nMAP", "NRE\nmedian"]
    #
    # path = res_path / "complete_outputs" / f"outputs_NBE [0].pt"
    # loss = torch.load(path, map_location=torch.device('cpu'))["final_test_losses"]
    # print(loss.shape)
    # if isinstance(loss, torch.Tensor):
    #     loss = loss.detach().cpu().numpy()
    # losses.append(loss)
    #
    # path = res_path / "complete_outputs" / f"outputs_NRE [0].pt"
    # loss = abs(torch.load(path, map_location=torch.device('cpu'))["true_all_theta"] - torch.load(path, map_location=torch.device('cpu'))["all_quantiles"][0.5])
    # print(loss.shape)
    # if isinstance(loss, torch.Tensor):
    #     loss = loss.detach().cpu().numpy()
    # losses.append(loss)
    #
    # path = res_path / "complete_outputs" / f"outputs_MLE [0].pt"
    # loss = abs(torch.load(path, map_location=torch.device('cpu'))["true_all_theta"] - torch.load(path, map_location=torch.device('cpu'))["final_all_theta"]* torch.sqrt(torch.tensor([1, 1 , 1/4, 1, 1])))
    # print(loss.shape)
    # if isinstance(loss, torch.Tensor):
    #     loss = loss.detach().cpu().numpy()
    #     loss[np.argmax(loss[:, 1]), 1] = baseline[1]
    #     loss[np.argmax(loss[:, 2]), 2] = baseline[2]
    # losses.append(loss)
    # print(loss.shape)
    # labels += ["MLE"]
    #
    # for i, name in enumerate(theta_names):
    #
    #     axs[i].boxplot(
    #         [loss[:,i] for loss in losses],
    #         tick_labels=labels,
    #         patch_artist=True,
    #         boxprops=dict(facecolor=THETA_COLORS[i]["test"], alpha=0.6),
    #         medianprops=dict(linestyle="-", linewidth=2, color="black"),
    #     )
    #     axs[i].tick_params(axis="x", labelrotation=45)
    #     axs[i].set_title("MAE – " + name)
    #     axs[i].set_ylabel("MAE" if i == 0 else "")
    #
    #     axs[i].axhline(
    #         baseline[i], color=THETA_COLORS[i]["baseline"], ls="--", label="baseline" # + name
    #     )
    #     if i == len(axs) - 1 :
    #         axs[i].legend()
    #
    # fig_box.suptitle(f"Test error") #  {trainers[0].test_dataset_hist[0].missing_ratio*100}%") # - ~80% missing values
    # fig_box.tight_layout()
    # plt.show()
    #
    # # fig_box.savefig(output_path / f"{model}-score_test_NBE-NRE-MLE.pdf", format="pdf", bbox_inches="tight")
    # # fig_box.savefig(output_path / f"{model}-score_test_NBE-NRE-MLE.png", format="png", bbox_inches="tight")
    #
    # # fig_box.savefig(output_path / f"{model}-score_test_median-MAP.pdf", format="pdf", bbox_inches="tight")
    # # fig_box.savefig(output_path / f"{model}-score_test_median-MAP.png", format="png", bbox_inches="tight")

    ######################
    ####### Missing ######
    ######################

    # print(res_path)
    #
    # fig_box, axs = plt.subplots(1, len(theta_names), figsize=(12, 5))
    #
    # method = "NBE"
    # losses = []
    # labels = ["~73%\nmissing\nratio", "~86%\nmissing\nratio", "~96%\nmissing\nratio"]
    # for j in range(len(labels)):
    #     path = (
    #         res_path
    #         / "complete_outputs"
    #         / f"outputs - {method} [{j}].pt"
    #     )
    #     loss = torch.load(path, map_location=torch.device('cpu'))["final_test_losses"]
    #     if isinstance(loss, torch.Tensor):
    #         loss = loss.detach().cpu().numpy()
    #     losses.append(loss)
    #
    # for i, name in enumerate(theta_names):
    #
    #     axs[i].boxplot(
    #         [loss[:,i] for loss in losses],
    #         tick_labels=labels,
    #         patch_artist=True,
    #         boxprops=dict(facecolor=THETA_COLORS[i]["test"], alpha=0.6),
    #         medianprops=dict(linestyle="-", linewidth=2, color="black"),
    #     )
    #     axs[i].tick_params(axis="x")# , labelrotation=45)
    #     axs[i].set_title("MAE – " + name)
    #     axs[i].set_ylabel("MAE" if i == 0 else "")
    #
    #     axs[i].axhline(
    #         baseline[i], color=THETA_COLORS[i]["baseline"], ls="--", label="baseline"  # + name
    #     )
    #     if i == len(axs) - 1 :
    #         axs[i].legend()
    #
    # fig_box.suptitle(f"Test error comparaison - {method}")
    # fig_box.tight_layout()
    # plt.show()
    #
    # fig_box.savefig(output_path / f"{model}-score_test_{method}.pdf", format="pdf", bbox_inches="tight")
    # fig_box.savefig(output_path / f"{model}-score_test_{method}.png", format="png", bbox_inches="tight")

    #################
    ##### Time ######
    #################

    # simu_times, train_times, test_times = [], [], []
    # labels = ["NBE", "NRE", "MLE"]
    #
    # simu_times = [22*60+17, 22*60+17, 0] # ["26m42s", "26m42s"]
    #
    # path = (
    #     res_path
    #     / "loss_evolution"
    #     / f"history - NBE [0].pt"
    # )
    # train_times.append(torch.load(path, map_location=torch.device('cpu'))["train_time"])
    # path = (
    #     res_path
    #     / "complete_outputs"
    #     / f"outputs - NBE [0].pt"
    # )
    # test_times.append(torch.load(path, map_location=torch.device('cpu'))["test_time"])
    #
    # path = (
    #     res_path
    #     / "loss_evolution"
    #     / f"history - NRE [0].pt"
    # )
    # train_times.append(torch.load(path, map_location=torch.device('cpu'))["train_time"])
    # path = (
    #     res_path
    #     / "complete_outputs"
    #     / f"outputs - NRE [0].pt"
    # )
    # test_times.append(torch.load(path, map_location=torch.device('cpu'))["test_time"] - torch.load(path, map_location=torch.device('cpu'))["map_time"])
    #
    # train_times.append(0)
    # path = res_path / "complete_outputs" / f"outputs - MLE [0].pt"
    # test_times.append(torch.load(path, map_location=torch.device("cpu"))["test_time"])
    #
    # x = np.arange(len(labels))
    # width = 0.3
    #
    # fig_time, ax = plt.subplots(figsize=(8, 5))
    # # cols = ["#7AA7F1", "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    # bar0 = ax.bar(x - width, simu_times, width, label='Simulation', color="#7AA7F1")
    # bar1 = ax.bar(x, train_times, width, label='Train', color="#785EF0")
    # bar2 = ax.bar(x + width, test_times, width, label='Test', color="#DC267F")
    #
    # for b, t in zip([bar0, bar1, bar2], [simu_times, train_times, test_times]):
    #     labels_time = [format_seconds(val) if val!=0 else "" for val in t]
    #     ax.bar_label(b, labels=labels_time, padding=3)
    #
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.tick_params(axis="x", labelrotation=45)
    #
    # ax.set_ylabel("Time")
    # ax.set_title("Computation time comparison")
    # ax.legend()
    #
    # ax.margins(y=0.2)
    #
    # plt.tight_layout()
    # plt.show()
    #
    # fig_time.savefig(output_path / f"{model}-times_NBE-NRE-MLE.pdf", format="pdf", bbox_inches="tight")
    # fig_time.savefig(output_path / f"{model}-times_NBE-NRE-MLE.png", format="png", bbox_inches="tight")

    ##################
    ###### Loss ######
    ##################

    # path = (
    #         res_path
    #         / "loss_evolution"
    #         / f"history - NBE [0].pt"
    #     )
    # all_loss_train = torch.load(path, map_location=torch.device('cpu'))["train_loss_history"]
    # all_loss_test = torch.load(path, map_location=torch.device('cpu'))["test_loss_history"]
    #
    # fig, ax = plt.subplots(
    #         1, len(theta_names), figsize=(14, 4), constrained_layout=True
    #     )
    #
    # for i, name in enumerate(theta_names):
    #     ax[i].plot(
    #         [l[i] for l in all_loss_train],
    #         THETA_COLORS[i]["train"],
    #         label="train " + name,
    #     )
    #     ax[i].plot(
    #         range(freq_epochs_test, nb_epochs + 1, freq_epochs_test),
    #         [l[i] for l in all_loss_test],
    #         THETA_COLORS[i]["test"],
    #         label="test " + name,
    #         linestyle=(5, (10, 2)),
    #     )
    #
    #     ax[i].axhline(
    #         baseline[i],
    #         color=THETA_COLORS[i]["baseline"],
    #         ls="--",
    #         label="baseline " + name,
    #     )
    #
    #     ax[i].set_title("Evolution of " + name)
    #     ax[i].set_xlabel("epochs")
    #     ax[i].set_ylabel("MAE")
    #     ax[i].legend()
    #
    # fig.suptitle(f"Loss evolutions of NBE")
    #
    # fig.savefig(
    #     output_path
    #     / f"{model}-train_loss_NBE.png",
    #     format="png",
    #     bbox_inches="tight",
    # )
    # fig.savefig(
    #     output_path
    #     / f"{model}-train_loss_NBE.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    # )
    #
    # plt.show()

    path = (
            res_path
            / "loss_evolution"
            / f"history - NRE [0].pt"
    )
    all_loss_train = torch.load(path, map_location=torch.device('cpu'))["train_loss_history"]

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 4), constrained_layout=True
    )

    ax.plot(
        all_loss_train,
        THETA_COLORS[5]["train"],
        # label="Binary Cross Entropy",
    )

    ax.set_xlabel("epochs")
    ax.set_ylabel("BCE")
    # ax.legend()

    fig.suptitle(f"Loss evolution of NRE")

    fig.savefig(
        output_path
        / f"{model}-train_loss_NRE.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()

    ######################
    ###### KL score ######
    ######################
    #
    # path = res_path / "complete_outputs" / f"outputs - NBE [0] (3).pt"
    # loss = torch.load(path, map_location=torch.device("cpu"))["final_test_losses"]
    # kl_score = torch.load(path, map_location=torch.device("cpu"))["kl_score"]
    #
    # # loss = torch.concatenate([loss, kl_score.unsqueeze(0).T], dim=1)
    # # cor = torch.corrcoef(torch.log(loss).T).numpy().round(3)
    # # names = theta_names + ["KL score (log)"]
    # #
    # # cols = [c["train"] for c in THETA_COLORS[:4]]
    # # CMAP = LinearSegmentedColormap.from_list("custom_data", cols, N=100)
    # #
    # # fig, ax = plt.subplots()
    # # im = ax.imshow(cor, cmap=CMAP)
    # #
    # # ax.set_xticks(range(len(names)), labels=names)
    # # ax.set_yticks(range(len(names)), labels=names)
    # #
    # # for i in range(len(names)):
    # #     for j in range(len(names)):
    # #         text = ax.text(j, i, cor[i, j], ha="center", va="center", color="w")
    # #
    # # ax.set_title("Loss/score matrix correlation")
    # # fig.tight_layout()
    # # plt.show()
    # #
    # # fig.savefig(
    # #     output_path / f"{model}-log-KL-score.pdf", format="pdf", bbox_inches="tight"
    # # )
    # # fig.savefig(
    # #     output_path / f"{model}-log-KL-score.png", format="png", bbox_inches="tight"
    # # )
    #
    #
    # fig, ax = plt.subplots()
    # plt.hist(torch.log(kl_score), color=cols[0])
    # ax.set_title("NBE KL score")
    # fig.tight_layout()
    # plt.show()
    #
    # fig.savefig(
    #     output_path / f"{model}-KL-hist.pdf", format="pdf", bbox_inches="tight"
    # )
    # fig.savefig(
    #     output_path / f"{model}-KL-hist.png", format="png", bbox_inches="tight"
    # )


    ########
    ## 3D ##
    ########

    # print(res_path)
    #
    # fig_box, axs = plt.subplots(1, len(theta_names), figsize=(15, 5))
    #
    # method = "NBE"
    # losses = []
    # labels = ["~92.0%\nmissing\nratio", "~97.6%\nmissing\nratio", "~99.4%\nmissing\nratio"]
    # for j in range(len(labels)):
    #     path = (
    #             res_path
    #             / "complete_outputs"
    #             / f"outputs - {method} [{j}].pt"
    #     )
    #     loss = torch.load(path, map_location=torch.device('cpu'))["final_test_losses"]
    #     if isinstance(loss, torch.Tensor):
    #         loss = loss.detach().cpu().numpy()
    #     losses.append(loss)
    #
    # for i, name in enumerate(theta_names):
    #
    #     axs[i].boxplot(
    #         [loss[:,i] for loss in losses],
    #         tick_labels=labels,
    #         patch_artist=True,
    #         boxprops=dict(facecolor=THETA_COLORS[i]["test"], alpha=0.6),
    #         medianprops=dict(linestyle="-", linewidth=2, color="black"),
    #     )
    #     axs[i].tick_params(axis="x")# , labelrotation=45)
    #     axs[i].set_title("MAE – " + name)
    #     axs[i].set_ylabel("MAE" if i == 0 else "")
    #
    #     axs[i].axhline(
    #         baseline[i], color=THETA_COLORS[i]["baseline"], ls="--", label="baseline"  # + name
    #     )
    #     if i == len(axs) - 1 :
    #         axs[i].legend()
    #
    # fig_box.suptitle(f"Test error comparaison - {method}")
    # fig_box.tight_layout()
    # plt.show()
    #
    # fig_box.savefig(output_path / f"{model}-score_test_{method}.pdf", format="pdf", bbox_inches="tight")
    # fig_box.savefig(output_path / f"{model}-score_test_{method}.png", format="png", bbox_inches="tight")

    ########################
    ###### pred score ######
    ########################

    # print(res_path)
    # error_names = ["Mean Squared Error", "Mean Absolute Error", "Negative Log Predictive Density"]
    #
    # fig_box, axs = plt.subplots(1, len(error_names), figsize=(12, 5))
    #
    # losses = []
    # labels = ["NBE", "NRE"]
    # # labels = ["NRE\nMAP", "NRE\nmedian"]
    #
    # path = res_path / "complete_outputs" / f"pred_scores_NBE [0].pt"
    # loss = torch.load(path, map_location=torch.device('cpu'))["prediction_scores"]
    # print(loss.shape)
    # if isinstance(loss, torch.Tensor):
    #     loss = loss.detach().cpu().numpy()
    # losses.append(loss)
    #
    # path = res_path / "complete_outputs" / f"pred_scores_NRE [0].pt"
    # loss = torch.load(path, map_location=torch.device('cpu'))["prediction_scores"]
    # print(loss.shape)
    # if isinstance(loss, torch.Tensor):
    #     loss = loss.detach().cpu().numpy()
    # losses.append(loss)
    #
    # path = res_path / "complete_outputs" / f"pred_scores_MLE [0].pt"
    # loss = torch.load(path, map_location=torch.device('cpu'))["prediction_scores"]
    # print(loss.shape)
    # if isinstance(loss, torch.Tensor):
    #     loss = loss.detach().cpu().numpy()
    #     loss[np.argmax(loss[:, 1]), 1] = baseline[1]
    #     loss[np.argmax(loss[:, 2]), 2] = baseline[2]
    # losses.append(loss)
    # print(loss.shape)
    # labels += ["MLE"]
    #
    # if "error_names" not in locals() or error_names is None:
    #     error_names = [f"Param_{j}" for j in range(losses[0].shape[1])]
    #
    # col_width = 24
    # header = f"{'Modèle':<15} | " + " | ".join(f"{name:^{col_width}}" for name in error_names)
    # separator = "-" * len(header)
    #
    # print(separator)
    # print(header)
    # print(separator)
    #
    # for l, loss in zip(labels, losses):
    #     if hasattr(loss, "cpu"):
    #         loss = loss.detach().cpu().numpy()
    #
    #     means = loss.mean(axis=0)
    #     stds = loss.std(axis=0)
    #
    #     row_cells = [f"{m:.3e} ± {s:.3e}" for m, s in zip(means, stds)]
    #     row_str = f"{l:<15} | " + " | ".join(f"{c:^{col_width}}" for c in row_cells)
    #     print(row_str)
    #
    # print(separator)
