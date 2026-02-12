import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from utils import map_deepset, baseline_mae, THETA_COLORS


# def train_test(architecture, optimizer, train_loader, test_loader, nb_epochs, freq_epochs_test, device, print_result=False):
#     all_loss_train = []
#     all_loss_test = []
#
#     loss_final_train = None
#     theta_final_test = None
#
#     t = tqdm(range(0, nb_epochs), total=nb_epochs)
#
#     time_train = time.time()
#     time_test = 0
#
#     architecture.train()
#
#     for epoch in t:
#
#         MAE = torch.nn.L1Loss(reduction="none")
#
#         losses_theta = []
#         full_losses = []
#
#         for i, (batched_data, theta, deepset) in enumerate(train_loader):
#
#             optimizer.zero_grad()
#
#             batched_data, theta, deepset = batched_data.to(device).float(), theta.to(device), deepset.to(device)
#
#             deepset_mapped, theta = map_deepset(deepset, theta)
#
#             output = architecture(batched_data, deepset_mapped)
#
#             loss_vec = MAE(output, theta)
#
#             loss = loss_vec.mean()
#             loss.backward()
#             optimizer.step()
#
#             losses_theta0.append(loss_vec[:, 0].mean().item())
#             losses_theta1.append(loss_vec[:, 1].mean().item())
#             losses_theta2.append(loss_vec[:, 2].mean().item())
#             losses_theta3.append(loss_vec[:, 3].mean().item())
#
#             if epoch == (nb_epochs - 1)::
#                 full_losses.append(loss_vec.detach().cpu())
#
#         mean = torch.tensor(
#             [np.mean(losses_theta0), np.mean(losses_theta1), np.mean(losses_theta2),
#              np.mean(losses_theta3)]).detach().cpu()
#         # std  = torch.tensor([np.std(losses_theta0),  np.std(losses_theta1)])
#
#         result = {
#             "mean": mean,
#             # "std": std
#         }
#
#         if save_full:
#             result["full"] = torch.cat(full_losses, dim=0)
#         t.set_description(f"Epoch {epoch + 1}/{nb_epochs}")
#         t.set_postfix(
#             loss=f"{losses["mean"][0]:.4f}, {losses["mean"][1]:.4f}, {losses["mean"][2]:.4f} and {losses["mean"][3]:.4f}")
#
#         all_loss_train.append(losses["mean"])
#
#         if "full" in losses:
#             loss_final_train = losses["full"]
#
#         if (epoch + 1) % freq_epochs_test == 0:
#
#             time_test = time.time()
#
#             for data_test, theta_test, deepset_test in test_loader:
#                 output = architecture(data_test.float(), torch.arange(len(theta_test)))
#                 loss_vec = torch.abs(output - theta_test)
#
#                 all_loss_test.append(loss_vec.mean(dim=0).detach().cpu())
#
#                 theta_final_test = output.detach().cpu()
#
#             time_test = time.time() - time_test
#             time_train -= time_test
#
#     time_train = time.time() - time_train
#
#     if print_result :
#         print(f"Time to train : {time_train:.2f}s")
#
#         print(
#             f"Train loss : "
#             f"adv {all_loss_train[-1][0].tolist()} | "
#             f"range time {all_loss_train[-1][1].tolist()} | "
#             f"range space {all_loss_train[-1][2].tolist()} | "
#             f"std {all_loss_train[-1][3].tolist()}"
#         )
#
#         print(
#             f"Test loss : "
#             f"adv {all_loss_test[-1][0].tolist()} | "
#             f"range time {all_loss_test[-1][1].tolist()} | "
#             f"range space {all_loss_test[-1][2].tolist()} | "
#             f"std {all_loss_test[-1][3].tolist()}"
#         )
#
#     return all_loss_train, time_train, loss_final_train, all_loss_test, time_test, theta_final_test


def train_test(architecture, optimizer, train_loader, test_loader, nb_epochs, freq_epochs_test, device,
               res_path, print_result=False):

    all_loss_train = []
    all_loss_test = []

    loss_final_train = None
    loss_final_test = None

    t = tqdm(range(nb_epochs), total=nb_epochs)

    time_train_start = time.time()
    time_test_accumulated = 0

    MAE = torch.nn.L1Loss(reduction="none")

    for epoch in t:
        architecture.train()

        epoch_batch_means = []

        epoch_full_errors = []

        for i, (batched_data, theta, deepset) in enumerate(train_loader):
            optimizer.zero_grad()

            batched_data = batched_data.to(device).float()
            theta = theta.to(device)
            deepset = deepset.to(device)

            deepset_mapped, theta = map_deepset(deepset, theta, device)

            output = architecture(batched_data, deepset_mapped)

            loss_vec = MAE(output, theta)

            loss = loss_vec.mean()
            loss.backward()
            optimizer.step()

            epoch_batch_means.append(loss_vec.mean(dim=0).detach().cpu())

            if epoch == nb_epochs - 1:
                epoch_full_errors.append(loss_vec.detach().cpu())

        mean_epoch = torch.stack(epoch_batch_means).mean(dim=0)
        all_loss_train.append(mean_epoch)

        if epoch == nb_epochs - 1:
            loss_final_train = torch.cat(epoch_full_errors, dim=0)

        loss_str = ", ".join([f"{l:.4f}" for l in mean_epoch])
        t.set_description(f"Epoch {epoch + 1}/{nb_epochs}")
        t.set_postfix_str(f"Loss: [{loss_str}]")

        if (epoch + 1) % freq_epochs_test == 0:
            start_test = time.time()
            architecture.eval()

            test_errors_list = []

            with torch.no_grad():
                for data_test, theta_test, deepset_test in test_loader:
                    data_test = data_test.to(device).float()
                    theta_test = theta_test.to(device)
                    deepset_test = deepset_test.to(device)

                    deepset_mapped_test, theta_test = map_deepset(deepset_test, theta_test, device)

                    output = architecture(data_test, deepset_mapped_test)

                    errors = torch.abs(output - theta_test)
                    test_errors_list.append(errors.detach().cpu())

            full_test_errors = torch.cat(test_errors_list, dim=0)

            all_loss_test.append(full_test_errors.mean(dim=0))

            loss_final_test = full_test_errors

            time_test_accumulated += (time.time() - start_test)

    total_time_train = (time.time() - time_train_start) - time_test_accumulated

    if print_result:
        print(f"\nTime to train : {total_time_train:.2f}s")

        last_train = all_loss_train[-1].tolist()
        print("Train loss per param : " + train_loader.dataset.model.to_title(last_train, alt_format=True))

        if all_loss_test:
            last_test = all_loss_test[-1].tolist()
            print("Test loss per param : " + test_loader.dataset.model.to_title(last_test, alt_format=True))

    results = {
        "train_loss_history": torch.stack(all_loss_train),  # (Nb_epochs, Nb_params)
        "test_loss_history": torch.stack(all_loss_test) if all_loss_test else None,  # (Nb_tests, Nb_params)
        "train_time": total_time_train,
        "test_time": time_test_accumulated,
        "final_train_losses": loss_final_train,  # (Size_Train, Nb_params)
        "final_test_losses": loss_final_test  # (Size_Test, Nb_params)
    }

    save_path = res_path / "loss_evolution" / f"{architecture.label}_history.pt"
    res_path.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path)

    _plot_loss(all_loss_train, all_loss_test, test_loader, res_path, nb_epochs, freq_epochs_test, architecture.label)

    torch.save(architecture.state_dict(), res_path / "models" / f"{architecture.label}_architecture.pth")

    # return results


def _plot_loss(all_loss_train, all_loss_test, test_loader, res_path, nb_epochs, freq_epochs_test, label):
    theta_names = test_loader.dataset.model.param_names

    fig, ax = plt.subplots(1, len(theta_names), figsize=(20, 7), constrained_layout=True)
    based, _ = baseline_mae(test_loader.dataset)

    for i, name in enumerate(theta_names):
        ax[i].plot([l[i] for l in all_loss_train], THETA_COLORS[i]['train'], label='train ' + name)
        ax[i].plot(range(freq_epochs_test, nb_epochs + 1, freq_epochs_test), [l[i] for l in all_loss_test],
                   THETA_COLORS[i]['test'], label='test ' + name)

        ax[i].axhline(based[i], color=THETA_COLORS[i]['baseline'], ls='--', label='baseline ' + name)

        ax[i].set_title('Evolution of ' + name)
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel('MAE')
        ax[i].legend()

    fig.suptitle('Loss evolutions of ' + label)

    fig.savefig(res_path / "loss_evolution" / f"{label}_train_evolution.png", format="png", bbox_inches="tight")
    fig.savefig(res_path / "loss_evolution" / f"{label}_train_evolution.pdf", format="pdf", bbox_inches="tight")