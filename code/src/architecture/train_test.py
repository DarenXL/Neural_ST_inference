import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmin

import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from utils import map_deepset, baseline_mae, THETA_COLORS

import torch
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, architecture, device, res_path, optimizer=None):

        self.architecture = architecture.to(device)
        self.device = device
        self.res_path = Path(res_path)

        if optimizer is None:
            if self.architecture.nls:
                self.optimizer = torch.optim.Adam(architecture.parameters(), lr=1e-3)
            else:
                self.optimizer = torch.optim.Adam(architecture.parameters(), lr=5e-5)

        self.theta_names = None
        self.simu_time = None

        self.train_loaders_hist = []
        self.test_loaders_hist = []

        (self.res_path / "loss_evolution").mkdir(parents=True, exist_ok=True)
        (self.res_path / "architectures").mkdir(parents=True, exist_ok=True)
        (self.res_path / "complete_outputs").mkdir(parents=True, exist_ok=True)

    def save_architecture(self, name="final"):
        path = self.res_path / "architectures" / f"{self.architecture.label}_{name}.pth"
        torch.save(self.architecture.state_dict(), path)

    def _save_history(self, history, suffix="history"):
        path = (
            self.res_path / "loss_evolution" / f"{self.architecture.label}_{suffix}.pt"
        )
        torch.save(history, path)

    def _save_results(self, outputs, suffix="outputs"):
        path = (
            self.res_path
            / "complete_outputs"
            / f"{self.architecture.label}_{suffix}.pt"
        )
        torch.save(outputs, path)

    def train(
        self,
        train_loader,
        nb_epochs,
        test_loader=None,
        freq_epochs_test=10,
        print_result=False,
    ):
        if len(self.train_loaders_hist) > 0 :
            assert train_loader.dataset.model.param_names == self.train_loaders_hist[0].dataset.model.param_names
        self.train_loaders_hist.append(train_loader)

        if self.architecture.nls:
            self.train_nls(train_loader, nb_epochs, print_result)
        else:
            self.train_generic(
                train_loader, test_loader, nb_epochs, freq_epochs_test, print_result
            )

    def test(self, test_loader, theta0=None, print_result=False):
        if len(self.test_loaders_hist) > 0 :
            assert (
                test_loader.dataset.model.param_names
                == self.test_loaders_hist[0].dataset.model.param_names
            )
        self.test_loaders_hist.append(test_loader)

        if self.architecture.nls:
            self.test_nls(test_loader, theta0, print_result)
        else:
            self.test_generic(test_loader, print_result)

    def train_generic(
        self, train_loader, test_loader, nb_epochs, freq_epochs_test, print_result=False
    ):

        all_loss_train = []
        all_loss_test = []

        loss_final_train = None
        loss_final_test = None

        t = tqdm(range(nb_epochs), total=nb_epochs)

        time_train_start = time.time()
        time_test_accumulated = 0

        MAE = torch.nn.L1Loss(reduction="none")

        for epoch in t:
            self.architecture.train()

            epoch_batch_means = []

            epoch_full_errors = []

            for i, (batched_data, theta, deepset) in enumerate(train_loader):
                self.optimizer.zero_grad()

                batched_data = batched_data.to(self.device).float()
                theta = theta.to(self.device)
                deepset = deepset.to(self.device)

                deepset_mapped, theta = map_deepset(deepset, theta, self.device)

                output = self.architecture(batched_data, deepset_mapped)

                loss_vec = MAE(output, theta)

                loss = loss_vec.mean()
                loss.backward()
                self.optimizer.step()

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
                self.architecture.eval()

                test_errors_list = []

                with torch.no_grad():
                    for data_test, theta_test, deepset_test in test_loader:
                        data_test = data_test.to(self.device).float()
                        theta_test = theta_test.to(self.device)
                        deepset_test = deepset_test.to(self.device)

                        deepset_mapped_test, theta_test = map_deepset(
                            deepset_test, theta_test, self.device
                        )

                        output = self.architecture(data_test, deepset_mapped_test)

                        errors = torch.abs(output - theta_test)
                        test_errors_list.append(errors.detach().cpu())

                full_test_errors = torch.cat(test_errors_list, dim=0)

                all_loss_test.append(full_test_errors.mean(dim=0))

                loss_final_test = full_test_errors

                time_test_accumulated += time.time() - start_test

        total_time_train = (time.time() - time_train_start) - time_test_accumulated

        if print_result:
            print(f"\nTime to train : {total_time_train:.2f}s")

            last_train = all_loss_train[-1].tolist()
            print(
                "Train loss per param : "
                + train_loader.dataset.model.to_title(last_train, alt_format=True)
            )

            if all_loss_test:
                last_test = all_loss_test[-1].tolist()
                print(
                    "Test loss per param : "
                    + test_loader.dataset.model.to_title(last_test, alt_format=True)
                )

        results = {
            "train_loss_history": torch.stack(all_loss_train),
            "test_loss_history": (torch.stack(all_loss_test) if all_loss_test else None),
            "train_time": total_time_train,
            "time_test_accumulated": time_test_accumulated,
            "final_train_losses": loss_final_train,
            "final_test_losses": loss_final_test,
        }

        self._save_history(results)
        self.save_architecture()

        theta_names = test_loader.dataset.model.param_names

        fig, ax = plt.subplots(
            1, len(theta_names), figsize=(20, 7), constrained_layout=True
        )
        based, _ = baseline_mae(test_loader.dataset)

        for i, name in enumerate(theta_names):
            ax[i].plot(
                [l[i] for l in all_loss_train],
                THETA_COLORS[i]["train"],
                label="train " + name,
            )
            ax[i].plot(
                range(freq_epochs_test, nb_epochs + 1, freq_epochs_test),
                [l[i] for l in all_loss_test],
                THETA_COLORS[i]["test"],
                label="test " + name,
            )

            ax[i].axhline(
                based[i],
                color=THETA_COLORS[i]["baseline"],
                ls="--",
                label="baseline " + name,
            )

            ax[i].set_title("Evolution of " + name)
            ax[i].set_xlabel("epochs")
            ax[i].set_ylabel("MAE")
            ax[i].legend()

        fig.suptitle("Loss evolutions of " + self.architecture.label)

        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"{self.architecture.label}_train_evolution.png",
            format="png",
            bbox_inches="tight",
        )
        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"{self.architecture.label}_train_evolution.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    def test_generic(self, loader_test, print_result=False):
        all_loss_test = []
        all_theta_final = []
        all_true_theta = []

        time_start = time.time()
        self.architecture.eval()

        model_ref = loader_test.dataset.model
        t = tqdm(
            loader_test,
            total=len(loader_test),
            desc=f"Inference {self.architecture.label}",
        )

        with torch.no_grad():
            for i, (data_test, theta_test, deepset_test) in enumerate(t):
                data_test = data_test.to(self.device).float()
                theta_test = theta_test.to(self.device)
                #deepset_test = deepset_test.to(self.device)

                # deepset_mapped, theta_target = map_deepset(
                #     deepset_test, theta_test, self.device
                # )
                output = self.architecture(data_test, torch.arange(len(deepset_test)))

                loss = torch.abs(output - theta_test).detach().cpu()
                theta_final = output.detach().cpu()

                all_loss_test.append(loss)
                all_theta_final.append(theta_final)
                all_true_theta.append(theta_test.detach().cpu())

                if i % 5 == 0:
                    current_mae = torch.cat(all_loss_test).mean().item()
                    t.set_postfix(Loss=f"{current_mae:.4f}")

        time_total = time.time() - time_start
        full_loss_tensor = torch.cat(all_loss_test, dim=0)
        all_theta_final = torch.cat(all_theta_final, dim=0)
        all_true_theta = torch.cat(all_true_theta, dim=0)

        if print_result:
            mean_loss = full_loss_tensor.mean(dim=0)
            print(f"Time to test : {time_total:.2f}s")
            print(
                f"Test Loss (MAE) per param: "
                + model_ref.to_title(mean_loss, alt_format=True)
            )

        results = {
            "final_test_losses": full_loss_tensor,  # (Nb_epochs, Nb_params)
            "test_time": time_total,
            "final_all_theta": all_theta_final,  # (Size_Test, Nb_params)
            "true_all_theta": all_true_theta,
        }
        self._save_results(results)

        # return full_loss_tensor, all_theta_final, time_total

    def train_nls(self, train_loader, nb_epochs, print_result=False):

        all_loss_train = []

        t = tqdm(range(nb_epochs), total=nb_epochs)

        time_train_start = time.time()

        CE = nn.CrossEntropyLoss()

        for epoch in t:
            self.architecture.train()

            epoch_batch = []

            for i, (batched_data, theta, classi) in enumerate(train_loader):
                self.optimizer.zero_grad()

                batched_data = batched_data.to(self.device).float()
                theta = theta.to(self.device)
                classi = classi.to(self.device)

                output = self.architecture(batched_data, theta=theta)

                loss = CE(output, classi).mean()

                loss.backward()
                self.optimizer.step()

                epoch_batch.append(loss.detach().cpu())

            mean_epoch = torch.stack(epoch_batch).mean()
            all_loss_train.append(mean_epoch)

            t.set_description(f"Epoch {epoch + 1}/{nb_epochs}")
            t.set_postfix_str(f"Loss: {mean_epoch.item():.3f}")

        total_time_train = time.time() - time_train_start

        if print_result:
            print(f"\nTime to train : {total_time_train:.2f}s")

            last_train = all_loss_train[-1].tolist()
            print("Train loss : " + str(last_train))

        results = {
            "train_loss_history": torch.stack(all_loss_train),  # (Nb_epochs, Nb_params)
            "train_time": total_time_train,
        }

        self._save_history(results)
        self.save_architecture()

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)
        ax.plot(all_loss_train, "forestgreen", linewidth=2.0)

        fig.suptitle("Loss evolutions of " + self.architecture.label)
        ax.set_xlabel("epochs")
        ax.set_ylabel("Cross Entropy")

        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"{self.architecture.label}_train_evolution.png",
            format="png",
            bbox_inches="tight",
        )
        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"{self.architecture.label}_train_evolution.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    def _nls_optim(self, data, theta0, activation=lambda x: x):

        def obj(theta):
            theta_t = activation(theta.unsqueeze(0).to(self.device))
            # theta_t = theta.unsqueeze(0).to(self.device)
            h = self.architecture(data, theta=theta_t)

            logL = h[:, 1] - h[:, 0]
            return logL

        res = torchmin.minimize(
            obj,
            theta0,
            method="bfgs",
            options={
                "max_iter": 500,
                "lr": 0.2,
            },
            return_all=True,
        )
        return activation(res.x), activation(res.allvecs[-2])

    def test_nls(self, loader_test, theta0=None, print_result=False):
        all_loss_test = []
        all_theta_final = []
        all_true_theta = []

        time_start = time.time()

        model_ref = loader_test.dataset.model
        lower_bound = torch.tensor(model_ref.lower_bound).to(self.device)
        upper_bound = torch.tensor(model_ref.upper_bound).to(self.device)

        if theta0 is None:
            theta0_latent = torch.zeros(len(model_ref.param_names)).to(self.device)
        else:
            theta0_latent = theta0

        activation = (
            lambda x: torch.sigmoid(x) * (upper_bound - lower_bound) + lower_bound
        )

        t = tqdm(loader_test, total=len(loader_test), desc="Inference NLS")

        for i, (data_batch, theta_batch, classi_batch) in enumerate(t):

            data_batch = data_batch.to(self.device).float()
            true_theta_batch = theta_batch.to(self.device)

            for j in range(data_batch.size(0)):

                if classi_batch[j] == 1:
                    continue

                data_single = data_batch[j].unsqueeze(0)
                true_theta_single = true_theta_batch[j]

                output, output_not_nan = self._nls_optim(
                    data_single, theta0_latent, activation
                )

                loss = torch.abs(output - true_theta_single).detach().cpu()
                theta_final = output.detach().cpu()

                if torch.isnan(loss).any():
                    loss = torch.abs(output_not_nan.cpu() - true_theta_single.cpu())
                    theta_final = output_not_nan.detach().cpu()

                all_loss_test.append(loss.unsqueeze(0))
                all_theta_final.append(theta_final.unsqueeze(0))
                all_true_theta.append(true_theta_single.detach().cpu().unsqueeze(0))

            if i % 5 == 0 and len(all_loss_test) > 0:
                current_mae = torch.cat(all_loss_test).mean().item()
                t.set_postfix(Loss=f"{current_mae:.4f}")

        time_total = time.time() - time_start

        full_loss_tensor = torch.cat(all_loss_test, dim=0)
        all_theta_final = torch.cat(all_theta_final, dim=0)
        all_true_theta = torch.cat(all_true_theta, dim=0)

        mean_loss = full_loss_tensor.mean(dim=0)
        std_loss = full_loss_tensor.std(dim=0)

        if print_result:
            print(f"Time to test : {time_total:.2f}s")

            print(
                f"Test Loss (MAE) per param: "
                + model_ref.to_title(mean_loss, alt_format=True)
            )

        results = {
            "final_test_losses": full_loss_tensor,  # (Nb_epochs, Nb_params)
            "test_time": time_total,
            "final_all_theta": all_theta_final,  # (Size_Test, Nb_params)
            "true_all_theta": all_true_theta,
        }
        self._save_results(results)

        # return full_loss_tensor, all_theta_final, time_total
