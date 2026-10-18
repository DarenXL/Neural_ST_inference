from neural_st_inference.utils import baseline_mae, THETA_COLORS
from neural_st_inference.estimator import BaseEstimator

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC, Predictive
from pyro.poutine import trace

import torch
import torch.nn as nn
import torchmin
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Trainer(BaseEstimator):
    def __init__(
        self,
        network,
        device,
        res_path,
        optimizer=None,
        quantiles=tuple(),
        quantile_networks=tuple(),
        quantile_optimizer=None,
        estimation_label=None,
    ):
        super().__init__(
            device=device,
            res_path=res_path,
            estimation_label=(
                network.label if (estimation_label is None) else estimation_label
            ),
            nls=network.nls,
        )

        self.network = network.to(device)

        if optimizer is None:
            lr = 1e-3 if self.nls else 5e-5
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        self.quantiles = quantiles
        self.lower_indices = sorted(
            [i for i, q in enumerate(quantiles) if q < 0.5],
            key=lambda i: quantiles[i],
            reverse=True,
        )
        self.upper_indices = sorted(
            [i for i, q in enumerate(quantiles) if q > 0.5], key=lambda i: quantiles[i]
        )

        self.quantile_network = quantile_networks
        if quantile_optimizer is None:
            self.quantile_optimizer = [
                torch.optim.Adam(qnn.parameters(), lr=1e-4)
                for qnn in self.quantile_network
            ]
        else:
            self.quantile_optimizer = quantile_optimizer

        (self.res_path / "loss_evolution").mkdir(parents=True, exist_ok=True)
        (self.res_path / "networks").mkdir(parents=True, exist_ok=True)

    def save_network(self, name=""):
        path = self.res_path / "networks" / f"{self.network.label}{name}.pth"
        torch.save(self.network.state_dict(), path)

    def save_quantile_network(self, name=""):
        for q, qnn in zip(self.quantiles, self.quantile_network):
            path = self.res_path / "networks" / f"q{q}_{qnn.label}{name}.pth"
            torch.save(qnn.state_dict(), path)

    def _save_history(self, history, prefix="history"):
        idx = len(self.train_dataset_hist) - 1
        filename = f"{prefix}_{self.network.label} [{idx}].pt"
        path = self.res_path / "loss_evolution" / filename
        torch.save(history, path)

    @staticmethod
    def _q_loss(output, target, tau):
        target = target.view_as(output)
        loss = torch.max(tau * (target - output), (1 - tau) * (output - target))
        return loss.mean()

    def train(
        self,
        train_loader,
        nb_epochs: int,
        test_loader=None,
        freq_epochs_test: int = 10,
    ):
        """ "
        Train the neural network.
        Test the neural network every freq_epochs_test epochs if not NLS case.
        keeps track of the datasets used.
        """
        if len(self.train_dataset_hist) > 0:
            assert (
                train_loader.dataset.theta_alt_names
                == self.train_dataset_hist[0].dataset.theta_alt_names
            )
        self.train_dataset_hist.append(train_loader.dataset)
        self.simu_time += train_loader.dataset.time_generation

        if self.network.nls:
            self.train_nre(train_loader, nb_epochs)
        else:
            self.train_nbe(train_loader, test_loader, nb_epochs, freq_epochs_test)

    def test(self, test_loader, mcmc=True):
        """ "
        Test the neural network.
        keeps track of the datasets used.
        """
        if len(self.test_dataset_hist) > 0:
            assert (
                test_loader.dataset.theta_alt_names
                == self.test_dataset_hist[0].theta_alt_names
            ), "loader's model do not correspond with the precedent ones"

        assert isinstance(
            test_loader.sampler, torch.utils.data.SequentialSampler
        ), "DataLoader must be shuffle=False"

        self.test_dataset_hist.append(test_loader.dataset)

        if self.network.nls:
            if mcmc :
                return self.test_nre_MCMC(test_loader)
            else :
                return self.test_nre_optim(test_loader)
        else:
            return self.test_nbe(test_loader)

    def train_nbe(
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

        try:

            for epoch in t:
                self.network.train()

                epoch_batch_means = []

                epoch_full_errors = []

                for i, (batched_data, theta) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    for opt in self.quantile_optimizer:
                        opt.zero_grad()

                    batched_data = batched_data.to(
                        self.device, non_blocking=True
                    ).float()
                    theta = theta.to(self.device, non_blocking=True)

                    base_output = self.network(batched_data)

                    loss_median = MAE(base_output, theta)

                    all_outputs = {}

                    current_out = base_output
                    for idx in self.lower_indices:
                        q_net = self.quantile_network[idx]
                        delta = torch.exp(q_net(batched_data))
                        current_out = current_out - delta
                        all_outputs[self.quantiles[idx]] = current_out

                    current_out = base_output
                    for idx in self.upper_indices:
                        q_net = self.quantile_network[idx]
                        delta = torch.exp(q_net(batched_data))
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
                        for data_test, theta_test in test_loader:
                            data_test = data_test.to(
                                self.device, non_blocking=True
                            ).float()
                            theta_test = theta_test.to(self.device, non_blocking=True)

                            output = self.network(data_test)

                            errors = torch.abs(output - theta_test)
                            test_errors_list.append(errors.detach())

                    full_test_errors = torch.cat(test_errors_list, dim=0).cpu()

                    all_loss_test.append(full_test_errors.mean(dim=0))

                    loss_final_test = full_test_errors

                    time_test_accumulated += time.time() - start_test

                if epoch % 20 == 0:
                    self.save_network()
                    self.save_quantile_network()

        except KeyboardInterrupt:
            self.save_network()
            self.save_quantile_network()

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

        theta_alt_names = test_loader.dataset.theta_alt_names

        fig, ax = plt.subplots(
            1, len(theta_alt_names), figsize=(20, 5), constrained_layout=True
        )
        based, _ = baseline_mae(test_loader.dataset)

        for i, name in enumerate(theta_alt_names):
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

    def test_nbe(self, loader_test):
        all_loss_test = []
        all_theta_final = []
        all_true_theta = []
        all_quantiles = {}

        for q in self.quantiles:
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
            for i, (data_test, theta_test) in enumerate(t):
                data_test = data_test.to(self.device, non_blocking=True).float()
                theta_test = theta_test.to(self.device, non_blocking=True)

                base_output = self.network(data_test)

                loss = torch.abs(base_output - theta_test).detach()
                theta_final = base_output.detach()

                current_out = base_output
                for idx in self.lower_indices:
                    q_net = self.quantile_network[idx]
                    delta = torch.exp(q_net(data_test))
                    current_out = current_out - delta
                    all_quantiles[self.quantiles[idx]].append(current_out)

                current_out = base_output
                for idx in self.upper_indices:
                    q_net = self.quantile_network[idx]
                    delta = torch.exp(q_net(data_test))
                    current_out = current_out + delta
                    all_quantiles[self.quantiles[idx]].append(current_out)

                all_loss_test.append(loss)
                all_theta_final.append(theta_final)
                all_true_theta.append(theta_test.detach())

                if i % 5 == 0:
                    current_mae = torch.cat(all_loss_test).mean().item()
                    t.set_postfix(Loss=f"{current_mae:.4f}")

        time_total = time.time() - time_start
        full_loss_tensor = torch.cat(all_loss_test, dim=0)
        all_theta_final = torch.cat(all_theta_final, dim=0)
        all_true_theta = torch.cat(all_true_theta, dim=0)
        for q in self.quantiles:
            all_quantiles[q] = torch.cat(all_quantiles[q], dim=0)

        results = {
            "final_all_theta": all_theta_final,  # (Size_Test, Nb_params)
            "true_all_theta": all_true_theta,
            "all_quantiles": all_quantiles,
            "theta_names": self.theta_names,
            "final_test_losses": full_loss_tensor,  # (Nb_epochs, Nb_params)
            "train_time": self.train_time,
            "test_time": time_total,
            "simu_time": self.simu_time,
            "total_time": self.train_time + time_total + self.simu_time,
            "estimation_label": self.estimation_label,
            "nls": self.nls,
        }

        self.test_time += time_total
        self._save_results(results)

        return results

    def train_nre(self, train_loader, nb_epochs):

        all_loss_train = []

        t = tqdm(range(nb_epochs), total=nb_epochs)

        time_train_start = time.time()

        CE = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()

        for epoch in t:
            self.network.train()

            epoch_batch = []

            for i, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                batched_data, theta, classi = batch

                batched_data = batched_data.to(self.device, non_blocking=True).float()
                theta = theta.to(self.device, non_blocking=True)
                classi = classi.float().to(self.device, non_blocking=True)

                # classi = classi.view(-1)
                # output = self.network(batched_data, theta=theta).view(-1)
                output = self.network(batched_data, theta=theta).squeeze()

                loss = CE(output, classi).mean()

                loss.backward()
                self.optimizer.step()

                epoch_batch.append(loss.detach())

            mean_epoch = torch.stack(epoch_batch).mean().cpu()
            all_loss_train.append(mean_epoch)

            t.set_description(f"Epoch {epoch + 1}/{nb_epochs}")
            t.set_postfix_str(f"Loss: {mean_epoch.item():.3f}")

        total_time_train = time.time() - time_train_start

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

    def _nre_optim(self, data, theta0, activation=lambda x: x):

        def obj(theta):
            theta_t = activation(theta.unsqueeze(0).to(self.device))
            # theta_t = theta.unsqueeze(0).to(self.device)
            h = self.network(data, theta=theta_t)
            logL = h  # h[:, 1] - h[:, 0]
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

    def test_nre_optim(self, loader_test, theta0=None):
        all_loss_test = []
        all_theta_final = []
        all_true_theta = []

        time_start = time.time()

        dataset_ref = loader_test.dataset
        lower_bound = torch.tensor(dataset_ref.model.lower_bound).to(self.device)
        upper_bound = torch.tensor(dataset_ref.model.upper_bound).to(self.device)

        if not (dataset_ref.dev is None):
            lower_bound = torch.cat(
                [
                    lower_bound,
                    torch.tensor(
                        (
                            [dataset_ref.noise]
                            if not (dataset_ref.noise is None)
                            else [0.3]
                        ),
                        device=self.device,
                    ),
                ]
            )
            upper_bound = torch.cat(
                [upper_bound, torch.tensor([dataset_ref.dev], device=self.device)]
            )

        if not (dataset_ref.noise is None):
            lower_bound = torch.cat(
                [lower_bound, torch.tensor([0], device=self.device)]
            )
            upper_bound = torch.cat(
                [upper_bound, torch.tensor([dataset_ref.noise], device=self.device)]
            )

        print(lower_bound, upper_bound)

        if theta0 is None:
            theta0_latent = (lower_bound + upper_bound) / 2
        else:
            theta0_latent = theta0

        activation = (
            lambda x: torch.sigmoid(x) * (upper_bound - lower_bound) + lower_bound
        )

        t = tqdm(loader_test, total=len(loader_test), desc="Inference NLS")

        for i, batch in enumerate(t):
            data_batch, theta_batch = batch

            data_batch = data_batch.to(self.device, non_blocking=True).float()
            true_theta_batch = theta_batch.to(self.device, non_blocking=True)

            for j in range(data_batch.size(0)):

                data_single = data_batch[j].unsqueeze(0)
                true_theta_single = true_theta_batch[j]

                output, output_not_nan = self._nre_optim(
                    data_single, theta0_latent, activation
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

        results = {
            "final_all_theta": all_theta_final,  # (Size_Test, Nb_params)
            "true_all_theta": all_true_theta,
            "theta_names": self.theta_names,
            "final_test_losses": full_loss_tensor,  # (Nb_epochs, Nb_params)
            "train_time": self.train_time,
            "test_time": time_total,
            "simu_time": self.simu_time,
            "total_time": self.train_time + time_total + self.simu_time,
            "estimation_label": self.estimation_label,
            "nls": self.nls,
        }
        self.test_time += time_total
        self._save_results(results)

        return results

    def _pyro_model(self, x_obs, lower_bound, upper_bound):
        batch_size = x_obs.shape[0]

        with pyro.plate("batch_dimensions", batch_size):
            theta = pyro.sample(
                "theta", dist.Uniform(lower_bound, upper_bound).to_event(1)
            )

        log_ratio = -self.network(x_obs, theta=theta).squeeze(-1)

        with pyro.plate("batch_factors", batch_size):
            pyro.factor("log_ratio_factor", log_ratio)

    def test_nre_MCMC(self, loader_test, num_samples=2000, warmup_steps=500):
        """https://pyro.ai/examples/predictive_deterministic.html"""
        all_samples = []
        all_true_theta = []
        all_theta_final = []
        all_quantiles = {}
        for q in {*self.quantiles, 0.5}:
            all_quantiles[q] = []

        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False

        dataset_ref = loader_test.dataset
        lower_bound = torch.tensor(dataset_ref.model.lower_bound).to(self.device)
        upper_bound = torch.tensor(dataset_ref.model.upper_bound).to(self.device)

        if not (dataset_ref.dev is None):
            lower_bound = torch.cat(
                [
                    lower_bound,
                    torch.tensor(
                        (
                            [dataset_ref.noise]
                            if not (dataset_ref.noise is None)
                            else [0.3]
                        ),
                        device=self.device,
                    ),
                ]
            )
            upper_bound = torch.cat(
                [upper_bound, torch.tensor([dataset_ref.dev], device=self.device)]
            )

        if not (dataset_ref.noise is None):
            lower_bound = torch.cat(
                [lower_bound, torch.tensor([0], device=self.device)]
            )
            upper_bound = torch.cat(
                [upper_bound, torch.tensor([dataset_ref.noise], device=self.device)]
            )

        kernel = NUTS(self._pyro_model, max_tree_depth=5, jit_compile=True)
        # kernel = HMC(self._pyro_model, num_steps=3, jit_compile=True)
        mcmc = MCMC(
            kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1
        )

        time_start = time.time()

        t = tqdm(loader_test, total=len(loader_test), desc="Inference NLS")

        for i, batch in enumerate(t):
            data_batch, theta_batch = batch
            data_batch = data_batch.to(self.device).float()

            batch_chains = []
            batch_log_ratios = []

            for chain_idx in range(3):
                pyro.clear_param_store()

                mcmc.run(data_batch, lower_bound, upper_bound)
                chain_samples = mcmc.get_samples()["theta"].detach().cpu()

                batch_chains.append(chain_samples.detach().cpu())

                torch.cuda.empty_cache()

            samples = torch.stack(batch_chains, dim=0)
            samples = samples.reshape(-1, samples.shape[2], samples.shape[3])

            all_samples.append(samples)
            all_true_theta.append(theta_batch.detach().cpu())

        time_total = time.time() - time_start

        all_samples = torch.cat(all_samples, dim=1)
        all_samples = all_samples.permute(1, 0, 2)
        for q in self.quantiles:
            all_quantiles[q] = torch.quantile(all_samples, q, dim=1)
        all_theta_final = all_quantiles[0.5]
        all_true_theta = torch.concatenate(all_true_theta, dim=0)
        final_test_losses = torch.abs(all_theta_final - all_true_theta)

        results = {
            "final_all_theta": all_theta_final,  # (Size_Test, Nb_params)
            "true_all_theta": all_true_theta,
            "all_quantiles": all_quantiles,
            "theta_names": self.theta_names,
            "final_test_losses": final_test_losses,  # (Nb_epochs, Nb_params)
            "train_time": self.train_time,
            "test_time": time_total,
            "simu_time": self.simu_time,
            "total_time": self.train_time + time_total + self.simu_time,
            "estimation_label": self.estimation_label,
            "nls": self.nls,
        }
        self.test_time += time_total
        self._save_results(results)

        return results
