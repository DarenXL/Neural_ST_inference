import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmin

import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from utils import map_deepset, baseline_mae, THETA_COLORS, unique_preserve_order
from scipy.optimize import minimize
from scipy.stats import chi2

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC, Predictive

import torch
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, network, device, res_path, optimizer=None, quantiles=tuple(), quantile_networks=tuple(), quantile_optimizer=None):

        self.network = network.to(device)
        self.device = device
        self.res_path = Path(res_path)

        self.estimation_label = self.network.label
        self.nls = self.network.nls

        if optimizer is None:
            if self.network.nls:
                self.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            else:
                self.optimizer = torch.optim.Adam(network.parameters(), lr=5e-5)

        self.quantiles = quantiles

        self.lower_indices = [i for i, q in enumerate(quantiles) if q < 0.5]
        self.upper_indices = [i for i, q in enumerate(quantiles) if q > 0.5]

        self.lower_indices.sort(key=lambda i: quantiles[i], reverse=True)
        self.upper_indices.sort(key=lambda i: quantiles[i])

        self.quantile_network = quantile_networks
        if quantile_optimizer is None:
            self.quantile_optimizer = [
                torch.optim.Adam(_.parameters(), lr=1e-4) for _ in self.quantile_network
            ]
        else:
            self.quantile_optimizer = quantile_optimizer

        self.theta_names = None
        self.simu_time = 0
        self.train_time = 0
        self.test_time = 0

        self.train_dataset_hist = []
        self.test_dataset_hist = []

        (self.res_path / "loss_evolution").mkdir(parents=True, exist_ok=True)
        (self.res_path / "networks").mkdir(parents=True, exist_ok=True)
        (self.res_path / "complete_outputs").mkdir(parents=True, exist_ok=True)

    def save_network(self, name=""):
        path = self.res_path / "networks" / f"{self.network.label}{name}.pth"
        torch.save(self.network.state_dict(), path)

    def save_quantile_network(self, name=""):
        for q, qnn in zip(self.quantiles, self.quantile_network):
            path = self.res_path / "networks" / f"q{q} - {qnn.label}{name}.pth"
            torch.save(qnn.state_dict(), path)

    def _save_history(self, history, prefix="history"):
        path = (
            self.res_path
            / "loss_evolution"
            / f"{prefix} - {self.network.label} [{len(self.train_dataset_hist)-1}].pt"
        )
        torch.save(history, path)

    def _save_results(self, outputs, prefix="outputs"):
        path = (
            self.res_path
            / "complete_outputs"
            / f"{prefix} - {self.network.label} [{len(self.test_dataset_hist)-1}].pt"
        )
        torch.save(outputs, path)

    @staticmethod
    def _q_loss(output, target, tau):
        target = target.view_as(output)
        loss = torch.max(tau * (target - output), (1 - tau) * (output - target))
        return loss.mean()

    def train(
        self,
        train_loader,
        nb_epochs:int,
        test_loader=None,
        freq_epochs_test:int=10,
        print_result:bool=False,
    ):
        """"
        Train the neural network.
        Test the neural network every freq_epochs_test epochs if not NLS case.
        keeps track of the datasets used.
        """
        if len(self.train_dataset_hist) > 0:
            assert (
                train_loader.dataset.model.param_names
                == self.train_dataset_hist[0].dataset.model.param_names
            )
        self.train_dataset_hist.append(train_loader.dataset)
        self.simu_time += train_loader.dataset.time_generation

        if self.network.nls:
            self.train_nls(train_loader, nb_epochs, print_result)
        else:
            self.train_quantiles(
                train_loader, test_loader, nb_epochs, freq_epochs_test, print_result
            )

    def test(self, test_loader, dataset_gridsearch=None, merge_test_deepset=False, print_result:bool=False):
        """"
        Test the neural network.
        keeps track of the datasets used.
        """
        if len(self.test_dataset_hist) > 0:
            assert (
                test_loader.dataset.model.param_names
                == self.test_dataset_hist[0].model.param_names
            ), "loader's model do not correspond with the precedent ones"

        assert isinstance(test_loader.sampler, torch.utils.data.SequentialSampler), "DataLoader must be shuffle=False"

        self.test_dataset_hist.append(test_loader.dataset)

        if self.network.nls:
            # return self.test_nls_grid_quantiles(test_loader, dataset_gridsearch=dataset_gridsearch, print_result=print_result)
            return self.test_nls_optim(test_loader, print_result=print_result)
            # return self.test_nls_MCMC(test_loader)
        else:
            return self.test_quantiles(test_loader, merge_test_deepset=merge_test_deepset)

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

        MAE = torch.nn.L1Loss(reduction="none").to(self.device)

        for epoch in t:
            self.network.train()

            epoch_batch_means = []

            epoch_full_errors = []

            for i, (batched_data, theta, deepset) in enumerate(train_loader):
                self.optimizer.zero_grad()

                batched_data = batched_data.to(self.device, non_blocking=True).float()
                theta = theta.to(self.device, non_blocking=True)
                deepset = deepset.to(self.device, non_blocking=True)

                deepset_mapped, theta = map_deepset(deepset, theta, self.device)

                output = self.network(batched_data, deepset_mapped)

                loss_vec = MAE(output, theta)

                loss = loss_vec.mean()
                loss.backward()
                self.optimizer.step()

                epoch_batch_means.append(loss_vec.mean(dim=0).detach()) ###?

                if epoch == nb_epochs - 1:
                    epoch_full_errors.append(loss_vec.detach())

            mean_epoch = torch.stack(epoch_batch_means).mean(dim=0)
            all_loss_train.append(mean_epoch.cpu())

            if epoch == nb_epochs - 1:
                loss_final_train = torch.cat(epoch_full_errors, dim=0)

            loss_str = ", ".join([f"{l:.4f}" for l in mean_epoch])
            t.set_description(f"Epoch {epoch + 1}/{nb_epochs}")
            t.set_postfix_str(f"Loss: [{loss_str}]")

            if (epoch + 1) % freq_epochs_test == 0:
                start_test = time.time()
                self.network.eval()

                test_errors_list = []

                with torch.no_grad():
                    for data_test, theta_test, deepset_test in test_loader:
                        data_test = data_test.to(self.device, non_blocking=True).float()
                        theta_test = theta_test.to(self.device, non_blocking=True)

                        # deepset_test = deepset_test.to(self.device)
                        # deepset_mapped_test, theta_test = map_deepset(
                        #     deepset_test, theta_test, self.device
                        # )
                        # output = self.network(data_test, deepset_mapped_test)

                        output = self.network(
                            data_test, torch.arange(len(deepset_test)).to(self.device)
                        )

                        errors = torch.abs(output - theta_test)
                        test_errors_list.append(errors.detach()) ###?

                full_test_errors = torch.cat(test_errors_list, dim=0)

                all_loss_test.append(full_test_errors.mean(dim=0).cpu())

                loss_final_test = full_test_errors

                time_test_accumulated += time.time() - start_test

        total_time_train = (time.time() - time_train_start) - time_test_accumulated

        if print_result:
            print(f"\nTime to train : {total_time_train:.2f}s")

            last_train = all_loss_train[-1]  # .tolist()
            print(
                "Train loss per param : "
                + train_loader.dataset.model.to_title(last_train, alt_format=True)
            )

            if all_loss_test:
                last_test = all_loss_test[-1]  # .tolist()
                print(
                    "Test loss per param : "
                    + test_loader.dataset.model.to_title(last_test, alt_format=True)
                )

        results = {
            "train_loss_history": torch.stack(all_loss_train),
            "test_loss_history": (
                torch.stack(all_loss_test) if all_loss_test else None
            ),
            "train_time": total_time_train,
            "time_test_accumulated": time_test_accumulated,
            "final_train_losses": loss_final_train,
            "final_test_losses": loss_final_test,
        }

        self._save_history(results)
        self.save_network()

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
                linestyle=(5, (10, 2)),
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

        fig.suptitle("Loss evolutions of " + self.network.label)

        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"train_evolution - {self.network.label}.png",
            format="png",
            bbox_inches="tight",
        )
        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"train_evolution - {self.network.label}.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    def train_quantiles(
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
            self.network.train()

            epoch_batch_means = []

            epoch_full_errors = []

            for i, (batched_data, theta, deepset) in enumerate(train_loader):
                self.optimizer.zero_grad()
                for opt in self.quantile_optimizer:
                    opt.zero_grad()

                batched_data = batched_data.to(self.device, non_blocking=True).float()
                theta = theta.to(self.device, non_blocking=True)
                deepset = deepset.to(self.device, non_blocking=True)

                deepset_mapped, theta = map_deepset(deepset, theta, self.device)

                base_output = self.network(batched_data, deepset_mapped)

                loss_median = MAE(base_output, theta)

                all_outputs = {}

                current_out = base_output
                for idx in self.lower_indices:
                    q_net = self.quantile_network[idx]
                    delta = torch.exp(q_net(batched_data, deepset_mapped))
                    current_out = current_out - delta
                    all_outputs[self.quantiles[idx]] = current_out

                current_out = base_output
                for idx in self.upper_indices:
                    q_net = self.quantile_network[idx]
                    delta = torch.exp(q_net(batched_data, deepset_mapped))
                    current_out = current_out + delta
                    all_outputs[self.quantiles[idx]] = current_out

                total_loss = loss_median.mean()
                for q_val, out in all_outputs.items():
                    total_loss += self._q_loss(out, theta, q_val)

                total_loss.backward()

                self.optimizer.step()
                for opt in self.quantile_optimizer:
                    opt.step()

                epoch_batch_means.append(loss_median.mean(dim=0).detach())

                if epoch == nb_epochs - 1:
                    epoch_full_errors.append(loss_median.detach())

            mean_epoch = torch.stack(epoch_batch_means).mean(dim=0).cpu()
            all_loss_train.append(mean_epoch)

            if epoch == nb_epochs - 1:
                loss_final_train = torch.cat(epoch_full_errors, dim=0)

            loss_str = ", ".join([f"{l:.4f}" for l in mean_epoch])
            t.set_description(f"Epoch {epoch + 1}/{nb_epochs}")
            t.set_postfix_str(f"Loss: [{loss_str}]")

            if (epoch + 1) % freq_epochs_test == 0:
                start_test = time.time()
                self.network.eval()

                test_errors_list = []

                with torch.no_grad():
                    for data_test, theta_test, deepset_test in test_loader:
                        data_test = data_test.to(self.device, non_blocking=True).float()
                        theta_test = theta_test.to(self.device, non_blocking=True)

                        # deepset_test = deepset_test.to(self.device)
                        # deepset_mapped_test, theta_test = map_deepset(
                        #     deepset_test, theta_test, self.device
                        # )
                        # output = self.network(data_test, deepset_mapped_test)

                        output = self.network(
                            data_test, torch.arange(len(deepset_test)).to(self.device)
                        )

                        errors = torch.abs(output - theta_test)
                        test_errors_list.append(errors.detach())

                full_test_errors = torch.cat(test_errors_list, dim=0).cpu()

                all_loss_test.append(full_test_errors.mean(dim=0))

                loss_final_test = full_test_errors

                time_test_accumulated += time.time() - start_test

        total_time_train = (time.time() - time_train_start) - time_test_accumulated

        if print_result:
            print(f"\nTime to train : {total_time_train:.2f}s")

            last_train = all_loss_train[-1]  # .tolist()
            print(
                "Train loss per param : "
                + train_loader.dataset.model.to_title(last_train, alt_format=True)
            )

            if all_loss_test:
                last_test = all_loss_test[-1]  # .tolist()
                print(
                    "Test loss per param : "
                    + test_loader.dataset.model.to_title(last_test, alt_format=True)
                )

        results = {
            "train_loss_history": torch.stack(all_loss_train),
            "test_loss_history": (
                torch.stack(all_loss_test) if all_loss_test else None
            ),
            "train_time": total_time_train,
            "time_test_accumulated": time_test_accumulated,
            "final_train_losses": loss_final_train,
            "final_test_losses": loss_final_test,
        }
        self.train_time += total_time_train

        self._save_history(results)
        self.save_network()
        self.save_quantile_network()

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
                linestyle=(5, (10, 2)),
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

        fig.suptitle("Loss evolutions of " + self.network.label)

        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"train_evolution - {self.network.label}.png",
            format="png",
            bbox_inches="tight",
        )
        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"train_evolution - {self.network.label}.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    def test_generic(self, loader_test, merge_test_deepset=False, print_result=False):
        all_loss_test = []
        all_theta_final = []
        all_true_theta = []

        time_start = time.time()
        self.network.eval()

        model_ref = loader_test.dataset.model
        t = tqdm(
            loader_test,
            total=len(loader_test),
            desc=f"Inference {self.network.label}",
        )

        with torch.no_grad():
            for i, (data_test, theta_test, deepset_test) in enumerate(t):
                data_test = data_test.to(self.device, non_blocking=True).float()
                theta_test = theta_test.to(self.device, non_blocking=True)

                if merge_test_deepset:
                    deepset_test = deepset_test.to(self.device, non_blocking=True)

                    deepset_mapped, theta_test = map_deepset(
                        deepset_test, theta_test, self.device
                    )
                    output = self.network(data_test, deepset_mapped)
                else :
                    output = self.network(data_test, torch.arange(len(deepset_test)).to(self.device))

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

    def test_quantiles(self, loader_test, merge_test_deepset=False, print_result=False):
        all_loss_test = []
        all_theta_final = []
        all_true_theta = []
        all_quantiles = {}
        for q in self.quantiles :
            all_quantiles[q] = []

        time_start = time.time()
        self.network.eval()

        model_ref = loader_test.dataset.model
        t = tqdm(
            loader_test,
            total=len(loader_test),
            desc=f"Inference {self.network.label}",
        )

        with torch.no_grad():
            for i, (data_test, theta_test, deepset_test) in enumerate(t):
                data_test = data_test.to(self.device, non_blocking=True).float()
                theta_test = theta_test.to(self.device, non_blocking=True)

                if merge_test_deepset:
                    deepset_test = deepset_test.to(self.device, non_blocking=True)

                    deepset_mapped, theta_test = map_deepset(
                        deepset_test, theta_test, self.device
                    )
                    base_output = self.network(data_test, deepset_mapped)
                else :
                    base_output = self.network(data_test, torch.arange(len(deepset_test)).to(self.device))

                loss = torch.abs(base_output - theta_test).detach()
                theta_final = base_output.detach()

                current_out = base_output
                for idx in self.lower_indices:
                    q_net = self.quantile_network[idx]
                    delta = torch.exp(q_net(data_test, torch.arange(len(deepset_test)).to(self.device)))
                    current_out = current_out - delta
                    all_quantiles[self.quantiles[idx]].append(current_out)

                current_out = base_output
                for idx in self.upper_indices:
                    q_net = self.quantile_network[idx]
                    delta = torch.exp(q_net(data_test, torch.arange(len(deepset_test)).to(self.device)))
                    current_out = current_out + delta
                    all_quantiles[self.quantiles[idx]].append(current_out)

                all_loss_test.append(loss)
                all_theta_final.append(theta_final)
                all_true_theta.append(theta_test.detach())

                # with torch.no_grad() :
                #     for k in range(len(data_test)):
                #         coords, y, mask_obs = loader_test.dataset.sample_to_coord(data_test[k].detach().cpu().numpy())
                #
                #         cov_true = model_ref.eval_cov(coords[~mask_obs], y[~mask_obs], space_dim=model_ref.space_dim, theta_values=theta_test[k].cpu().numpy())
                #         cov_est = model_ref.eval_cov(coords[~mask_obs], y[~mask_obs], space_dim=model_ref.space_dim, theta_values=theta_final[k].cpu().numpy())
                #
                #         vals_true = cov_true.getValues()
                #         vals_true = np.array(vals_true).reshape(cov_true.getNRows(), cov_true.getNCols())
                #
                #         vals_est = cov_est.getValues()
                #         vals_est = np.array(vals_est).reshape(cov_est.getNRows(), cov_est.getNCols())
                #
                #         _, logdet_true = np.linalg.slogdet(vals_true)
                #         _, logdet_est = np.linalg.slogdet(vals_est)
                #         term_det = logdet_est - logdet_true
                #
                #         term_trace = np.trace(np.linalg.solve(vals_est, vals_true))
                #
                #         # Calcul final du KL score
                #         kl_score = 0.5 * (term_det + term_trace - len(vals_true))
                #
                #         print("KL score :", kl_score)
                #         print("MAE loss :", loss[k])

                if i % 5 == 0:
                    current_mae = torch.cat(all_loss_test).mean().item()
                    t.set_postfix(Loss=f"{current_mae:.4f}")

        time_total = time.time() - time_start
        full_loss_tensor = torch.cat(all_loss_test, dim=0)
        all_theta_final = torch.cat(all_theta_final, dim=0)
        all_true_theta = torch.cat(all_true_theta, dim=0)
        for q in self.quantiles :
            all_quantiles[q] = torch.cat(all_quantiles[q], dim=0)

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
            "all_quantiles":all_quantiles
        }
        self.test_time += time_total
        self._save_results(results)

        return results

    def train_nls(self, train_loader, nb_epochs, print_result=False):

        all_loss_train = []

        t = tqdm(range(nb_epochs), total=nb_epochs)

        time_train_start = time.time()

        CE = nn.BCEWithLogitsLoss() # nn.CrossEntropyLoss()

        for epoch in t:
            self.network.train()

            epoch_batch = []

            for i, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                if self.network.nls_sim :
                    batched_data, theta, deepset, classi = batch
                    deepset = deepset.to(self.device, non_blocking=True)
                    _, deepset_mapped = torch.unique_consecutive(
                        deepset, return_inverse=True, dim=0
                    )
                else :
                    batched_data, theta, classi = batch
                    deepset_mapped=None

                batched_data = batched_data.to(self.device, non_blocking=True).float()
                theta = theta.to(self.device, non_blocking=True)
                classi = classi.float().to(self.device, non_blocking=True)

                output = self.network(batched_data, deepset=deepset_mapped, theta=theta).squeeze()

                loss = CE(output, classi).mean()

                loss.backward()
                self.optimizer.step()

                epoch_batch.append(loss.detach())

            mean_epoch = torch.stack(epoch_batch).mean().cpu()
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
        self.train_time += total_time_train

        self._save_history(results)
        self.save_network()

        fig, ax = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)
        ax.plot(all_loss_train, "#DC267F", linewidth=2.0)

        fig.suptitle("Loss evolutions of " + self.network.label)
        ax.set_xlabel("epochs")
        ax.set_ylabel("Cross Entropy")

        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"train_evolution - {self.network.label}.png",
            format="png",
            bbox_inches="tight",
        )
        fig.savefig(
            self.res_path
            / "loss_evolution"
            / f"train_evolution - {self.network.label}.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    def _nls_optim(self, data, theta0, activation=lambda x: x, deepset=None):

        def obj(theta):
            theta_t = activation(theta.unsqueeze(0).to(self.device))
            # theta_t = theta.unsqueeze(0).to(self.device)
            h = self.network(data, deepset, theta=theta_t)
            logL = h # h[:, 1] - h[:, 0]
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

    def test_nls_optim(self, loader_test, theta0=None, print_result=False):
        all_loss_test = []
        all_theta_final = []
        all_true_theta = []

        time_start = time.time()

        model_ref = loader_test.dataset.model
        lower_bound = torch.tensor(model_ref.lower_bound).to(self.device)
        upper_bound = torch.tensor(model_ref.upper_bound).to(self.device)

        if theta0 is None:
            theta0_latent = (lower_bound + upper_bound)/2
        else:
            theta0_latent = theta0

        activation = (
            lambda x: torch.sigmoid(x) * (upper_bound - lower_bound) + lower_bound
        )

        t = tqdm(loader_test, total=len(loader_test), desc="Inference NLS")

        for i, batch in enumerate(t):
            if self.network.nls_sim :
                data_batch, theta_batch, deepset_batch, classi_batch = batch
                deepset_batch = unique_preserve_order(deepset_batch)
            else :
                data_batch, theta_batch, classi_batch = batch
                deepset_batch = None

            data_batch = data_batch.to(self.device, non_blocking=True).float()
            true_theta_batch = theta_batch.to(self.device, non_blocking=True)

            for j in range(data_batch.size(0)):

                if classi_batch[j] == 1:
                    continue

                data_single = data_batch[j].unsqueeze(0)
                if not(deepset_batch is None):
                    deepset_single = deepset_batch[j].unsqueeze(0)
                else :
                    deepset_single = None
                true_theta_single = true_theta_batch[j]

                output, output_not_nan = self._nls_optim(
                    data_single, theta0_latent, activation, deepset=deepset_single
                )

                loss = torch.abs(output - true_theta_single).detach()
                theta_final = output.detach()

                if torch.isnan(loss).any():
                    loss = torch.abs(output_not_nan - true_theta_single)
                    theta_final = output_not_nan.detach()

                all_loss_test.append(loss.unsqueeze(0))
                all_theta_final.append(theta_final.unsqueeze(0))
                all_true_theta.append(true_theta_single.detach().unsqueeze(0))

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
        self.test_time += time_total
        self._save_results(results)

        # return full_loss_tensor, all_theta_final, time_total

    def _pyro_model(self, x_obs, lower_bound, upper_bound):
        batch_size = x_obs.shape[0]

        with pyro.plate("batch_dimensions", batch_size):
            theta = pyro.sample("theta", dist.Uniform(lower_bound, upper_bound).to_event(1))

        # with torch.no_grad():
        log_ratio = -self.network(x_obs, theta=theta).squeeze(-1)

        with pyro.plate("batch_factors", batch_size):
            pyro.factor("log_ratio_factor", log_ratio)

        # pyro.deterministic("log_ratio_samples", log_ratio)

    def test_nls_MCMC(self, loader_test, num_samples=1000, step_size=0.05, burn_in=200):
        """https://pyro.ai/examples/predictive_deterministic.html"""
        all_samples = []
        all_true_theta = []
        all_theta_final = []
        all_quantiles = {}
        for q in self.quantiles :
            all_quantiles[q] = []

        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False

        model_ref = loader_test.dataset.model
        lower_bound = torch.tensor(model_ref.lower_bound).to(self.device)
        upper_bound = torch.tensor(model_ref.upper_bound).to(self.device)

        num_samples = 2000
        warmup_steps = 500

        kernel = NUTS(self._pyro_model, max_tree_depth=5, jit_compile=True)
        # kernel = HMC(self._pyro_model, num_steps=3, jit_compile=True)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)

        time_start = time.time()
        time_map = 0

        t = tqdm(loader_test, total=len(loader_test), desc="Inference NLS")

        for i, batch in enumerate(t):
            data_batch, theta_batch, _ = batch
            data_batch = data_batch.to(self.device).float()

            # mcmc.run(data_batch, lower_bound, upper_bound)
            # samples = mcmc.get_samples()["theta"]

            batch_chains = []
            batch_log_ratios = []

            for chain_idx in range(3):
                mcmc.run(data_batch, lower_bound, upper_bound)
                samples = mcmc.get_samples()
                chain_samples = samples["theta"]

                # determinist = Predictive(self._pyro_model, samples)(data_batch, lower_bound, upper_bound)
                # chain_log_ratio = determinist["log_ratio_samples"].cpu()

                with torch.no_grad():
                    time_map_t = time.time()
                    log_ratio_list = []

                    for s_idx in range(chain_samples.size(0)):
                        single_theta = chain_samples[s_idx]

                        log_ratio = -self.network(
                            data_batch, theta=single_theta
                        ).squeeze(-1)

                        log_ratio_list.append(log_ratio.cpu())

                    chain_log_ratio = torch.stack(log_ratio_list)
                    time_map += time.time() - time_map_t

                batch_chains.append(chain_samples.cpu())
                batch_log_ratios.append(chain_log_ratio)

            samples = torch.stack(batch_chains, dim=0)
            samples = samples.reshape(-1, samples.shape[2], samples.shape[3])

            log_ratios = torch.stack(batch_log_ratios, dim=0)
            print(log_ratios.shape)
            log_ratios = log_ratios.reshape(-1, log_ratios.shape[-1])
            print(log_ratios.shape)
            map_indices = torch.argmax(log_ratios, dim=0)

            print(map_indices.shape)

            theta_MAP = samples[map_indices, torch.arange(data_batch.size(0)), :]

            print(theta_MAP.shape)

            all_theta_final.append(theta_MAP)

            all_samples.append(samples)
            all_true_theta.append(theta_batch.cpu())

        time_total = time.time() - time_start

        all_samples = torch.cat(all_samples, dim=1)
        all_samples = all_samples.permute(1, 0, 2)
        all_theta_final = torch.concatenate(all_theta_final, dim=0) #  all_theta_final = torch.median(all_samples, dim=1)
        all_true_theta = torch.concatenate(all_true_theta, dim=0)
        final_test_losses = torch.abs(all_theta_final - all_true_theta)
        for q in self.quantiles:
            all_quantiles[q] = torch.quantile(all_samples, q, dim=1)

        results = {
            "final_test_losses": final_test_losses,  # (Nb_epochs, Nb_params)
            "test_time": time_total-time_map,
            "map_time":time_map,
            "final_all_theta": all_theta_final,  # (Size_Test, Nb_params)
            "true_all_theta": all_true_theta,
            "all_quantiles": all_quantiles,
        }
        self.test_time += time_total
        self._save_results(results)

        return results

    def test_nls_grid(self, loader_test, dataset_gridsearch, print_result=False):

        with torch.no_grad():

            all_loss_test = []
            all_theta_final = []
            all_true_theta = []

            model_ref = loader_test.dataset.model
            simu_nls = self.network.nls_sim # self.train_dataset_hist[-1].simu_nls

            time_start = time.time()

            t = tqdm(loader_test, total=len(loader_test)//2, desc="Inference NLS")

            for i, batch in enumerate(t):
                data_batch, theta_batch, _ = batch
                deepset_batch = torch.arange(len(data_batch),dtype=int).to(self.device, non_blocking=True)

                data_batch = data_batch.to(self.device, non_blocking=True).float()
                theta_batch = theta_batch.to(self.device, non_blocking=True)

                batch_size = data_batch.shape[0]
                max_logl = torch.full((batch_size,), float("-inf"), device=self.device)
                max_theta = torch.zeros((batch_size, dataset_gridsearch.get_theta_tensor().shape[1]), dtype=torch.float64, device=self.device)

                for k in range(len(dataset_gridsearch)):  # //2 if N

                    current_theta = dataset_gridsearch.get_theta_tensor()[k].to(self.device, non_blocking=True)

                    if simu_nls:
                        simu, _ = dataset_gridsearch.get_grid(
                            k, deepset_size=dataset_gridsearch.max_deepset_size
                        )
                        simu = torch.from_numpy(simu).to(self.device, non_blocking=True)
                        simu = torch.concatenate((simu, torch.ones_like(simu)), dim=1)
                        # simu = simu.repeat(batch_size, *([1] * (simu.ndim -1)))
                        simu = simu.unsqueeze(0).expand(batch_size, *([-1] * (simu.ndim))).reshape(-1, *simu.shape[1:])
                        h = self.network(
                            data_batch,
                            deepset=deepset_batch.repeat_interleave(dataset_gridsearch.max_deepset_size),
                            theta=simu,
                        )
                    else:
                        h = self.network(
                            data_batch,
                            theta=current_theta.unsqueeze(0).repeat((batch_size,1)),
                        )

                    logl = (h[:, 0] - h[:, 1]).detach()
                    mask = logl > max_logl
                    max_logl[mask] = logl[mask]
                    max_theta[mask] = current_theta.detach()

                loss = torch.abs(max_theta - theta_batch).detach()
                theta_final = max_theta.detach()

                all_loss_test.append(loss)
                all_theta_final.append(theta_final)
                all_true_theta.append(theta_batch.detach())

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

            return results

    def test_nls_grid_quantiles(self, loader_test, dataset_gridsearch, print_result=False):

        plt.show()

        with torch.no_grad():

            all_loss_test = []
            all_theta_final = []
            all_true_theta = []
            all_max_logl = []

            all_mask_quantile = {}
            count_true_quantile = {}
            for q in self.quantiles:
                all_mask_quantile[q] = []
                count_true_quantile[q] = 0

            model_ref = loader_test.dataset.model
            simu_nls = self.network.nls_sim  # self.train_dataset_hist[-1].simu_nls
            print("simu_nls", simu_nls)

            time_start = time.time()

            t = tqdm(loader_test, total=len(loader_test) // 2, desc="Inference NLS")

            for i, batch in enumerate(t):
                data_batch, theta_batch, _ = batch
                deepset_batch = torch.arange(len(data_batch), dtype=int).to(self.device, non_blocking=True)

                data_batch = data_batch.to(self.device, non_blocking=True).float()
                theta_batch = theta_batch.to(self.device, non_blocking=True)

                batch_size = data_batch.shape[0]
                max_logl = torch.full((batch_size,), float("-inf"), device=self.device)
                max_theta = torch.zeros(
                    (batch_size, dataset_gridsearch.get_theta_tensor().shape[1]),
                    dtype=torch.float64,
                    device=self.device,
                )

                surface = []

                for k in range(len(dataset_gridsearch)):  # //2 if N

                    current_theta = dataset_gridsearch.get_theta_tensor()[k].to(
                        self.device
                    )

                    if simu_nls:
                        simu, _ = dataset_gridsearch.get_grid(
                            k, deepset_size=dataset_gridsearch.max_deepset_size
                        )
                        simu = torch.from_numpy(simu).to(self.device)
                        simu = torch.concatenate((simu, torch.ones_like(simu)), dim=1)
                        # simu = simu.repeat(batch_size, *([1] * (simu.ndim -1)))
                        simu = (
                            simu.unsqueeze(0)
                            .expand(batch_size, *([-1] * (simu.ndim)))
                            .reshape(-1, *simu.shape[1:])
                        )
                        h = self.network(
                            data_batch,
                            deepset=deepset_batch.repeat_interleave(
                                dataset_gridsearch.max_deepset_size
                            ),
                            theta=simu,
                        )
                    else:
                        h = self.network(
                            data_batch,
                            theta=current_theta.unsqueeze(0).repeat((batch_size, 1)),
                        )

                    logl = -h.squeeze(0).detach() # (h[:, 0] - h[:, 1]).detach()
                    surface.append(logl)

                    mask = logl > max_logl
                    max_logl[mask] = logl[mask]
                    max_theta[mask] = current_theta.detach()

                loss = torch.abs(max_theta - theta_batch).detach()
                theta_final = max_theta.detach()

                all_max_logl.append(max_logl.detach())
                all_loss_test.append(loss)
                all_theta_final.append(theta_final)
                all_true_theta.append(theta_batch.detach())

                surface = torch.cat(surface, dim=0)
                # for q in self.quantiles:
                #     mask_quantile = 2 * (
                #         max_logl.detach().cpu() - surface.detach().cpu()
                #     ) <= chi2.isf(1 - q, len(theta_batch[0]))
                #
                #     all_mask_quantile[q].append(mask_quantile)
                #
                # h_true_theta = self.network(
                #             data_batch,
                #             theta=theta_batch,
                #         )
                # logl_true_theta = -h_true_theta.detach()
                #
                # for q in self.quantiles:
                #     count_true_quantile[q] += (2 * (
                #         max_logl.detach().cpu() - logl_true_theta.detach().cpu()
                #     )) <= chi2.isf(1 - q, len(theta_batch[0]))

                # plt.scatter(dataset_gridsearch.get_theta_tensor()[:,0],dataset_gridsearch.get_theta_tensor()[:,1], c=surface)
                # # plt.scatter(
                # #     dataset_gridsearch.get_theta_tensor()[mask_quantile, 0],
                # #     dataset_gridsearch.get_theta_tensor()[mask_quantile, 1],
                # #     edgecolors="black",
                # #     linewidths=1,
                # #     facecolors="none",
                # #     s=50,
                # #     label="Quantile",
                # # )
                # plt.scatter(theta_batch[0,0],theta_batch[0,1],c='black',marker='x')
                # plt.scatter(theta_final[0,0], theta_final[0,1], c="red", marker="x")
                # plt.show()

            time_total = time.time() - time_start

            full_loss_tensor = torch.cat(all_loss_test, dim=0)
            all_theta_final = torch.cat(all_theta_final, dim=0)
            all_true_theta = torch.cat(all_true_theta, dim=0)
            all_max_logl = torch.cat(all_max_logl, dim=0)

            # for q in self.quantiles:
            #     all_mask_quantile[q] = torch.cat(all_mask_quantile[q], dim=0)
            #     print(q,count_true_quantile[q], count_true_quantile[q]/len(all_true_theta))
            #     print(all_mask_quantile[q].sum())

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
                "all_max_logl":all_max_logl,
            }
            self.test_time += time_total
            self._save_results(results)

            return results
