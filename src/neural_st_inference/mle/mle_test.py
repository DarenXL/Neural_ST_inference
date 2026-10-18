from neural_st_inference.estimator import BaseEstimator
from neural_st_inference.models.utils_gp import train_gneiting_gp


import torch
import time
from tqdm import tqdm
import numpy as np


class MLE(BaseEstimator):
    def __init__(self, device, res_path, estimation_label=None):
        super().__init__(
            device=device,
            res_path=res_path,
            estimation_label="MLE" if (estimation_label is None) else estimation_label,
            nls=True,
        )

    def test_mle(self, dataset):
        # dataset = test_loader.dataset
        self.test_dataset_hist.append(dataset)

        all_loss_test = []
        all_theta_final = []
        all_true_theta = []

        time_start = time.time()

        print("len data", len(dataset))

        for i in tqdm(range(len(dataset)), desc=f"Inference {self.estimation_label}"):
            data_test, theta_test = dataset[i]

            data_test = data_test.detach().float()
            theta_test = theta_test

            for j in range(len(data_test)):

                print(j, "Truth :", theta_test)

                coords, y, mask = dataset.sample_to_coord(data_test[j], shuffle=300)
                coords = coords.flip(1)

                model_stationary, lik_stationary, losses_stationary = train_gneiting_gp(
                    train_x=torch.tensor(coords).detach().float(),
                    train_y=torch.tensor(y).detach().float(),
                    spatial_dim=1,
                    nu=1.5,
                    n_iter=600,
                    lr=0.05,
                )
                covar_module = model_stationary.covar_module
                k = covar_module.base_kernel
                output = torch.tensor(
                    [
                        k.adv.item(),
                        k.length_scale_t.item(),
                        k.length_scale_s.item(),
                        torch.sqrt(covar_module.outputscale).item(),
                        torch.sqrt(lik_stationary.noise).item(),
                    ]
                )

                loss = torch.abs(output - theta_test[j])

                all_loss_test.append(loss)
                all_theta_final.append(output)
                all_true_theta.append(theta_test[j])

        time_total = time.time() - time_start
        full_loss_tensor = torch.stack(all_loss_test, dim=0)
        all_theta_final = torch.stack(all_theta_final, dim=0)
        all_true_theta = torch.stack(all_true_theta, dim=0)

        results = {
            "final_all_theta": (
                all_theta_final.detach().cpu()
                if hasattr(all_theta_final, "detach")
                else all_theta_final
            ),
            "true_all_theta": (
                all_true_theta.detach().cpu()
                if hasattr(all_true_theta, "detach")
                else all_true_theta
            ),
            "theta_names": self.theta_names,
            "final_test_losses": (
                full_loss_tensor.detach().cpu()
                if hasattr(full_loss_tensor, "detach")
                else full_loss_tensor
            ),
            "simu_time": self.simu_time,
            "train_time": self.train_time,
            "test_time": time_total,
            "total_time": self.simu_time + self.train_time + time_total,
            "estimation_label": self.estimation_label,
        }
        self.test_time += time_total
        self._save_results(results)

        return results
