import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmin

import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from utils import map_deepset, baseline_mae, THETA_COLORS, unique_preserve_order
from scipy.optimize import minimize

import torch
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

class MLE:
    def __init__(self, device, res_path):

        self.device = device
        self.res_path = Path(res_path)

        self.estimation_label = "MLE"
        self.nls = True # To avoid problem in graphs

        self.theta_names = None
        self.simu_time = 0
        self.train_time = 0
        self.test_time = 0

        self.train_dataset_hist = []
        self.test_dataset_hist = []

        (self.res_path / "complete_outputs").mkdir(parents=True, exist_ok=True)

    def _save_results(self, outputs, prefix="outputs"):
        path = (
            self.res_path
            / "complete_outputs"
            / f"{prefix} - MLE [{len(self.test_dataset_hist)-1}].pt"
        )
        torch.save(outputs, path)

    def mle(self, likelihood, coords, y, theta0, bounds=None):

        def obj(theta):
            res = -likelihood(coords, y, theta)
            print(" " ,res, " | ", theta)
            return res

        res = minimize(
            obj, theta0, method="Nelder-Mead", bounds=bounds, options={"maxiter": 2000, 'disp': False} # "Powell"
        )
        return res.x, res

    def test_mle(self, dataset):
        # dataset = test_loader.dataset
        self.test_dataset_hist.append(dataset)

        all_loss_test = []
        all_theta_final = []
        all_true_theta = []

        time_start = time.time()

        # t = tqdm(
        #     dataset,
        #     total=len(dataset),
        #     desc=f"Inference {self.estimation_label}",
        # )

        print("len data", len(dataset))

        lower_bound = torch.tensor(dataset.model.lower_bound)
        upper_bound = torch.tensor(dataset.model.upper_bound)
        theta0 = (lower_bound + upper_bound)/2
        bounds = torch.stack((lower_bound, upper_bound)).T
        print(bounds)

        with torch.no_grad():
            for i in tqdm(
                range(len(dataset)), desc=f"Inference {self.estimation_label}"
            ):
                data_test, theta_test, _ = dataset[i]

                data_test = data_test.detach().numpy()
                theta_test = theta_test

                for j in range(len(data_test)):

                    print(j, "Truth :", theta_test)

                    coords, y, mask = dataset.sample_to_coord(data_test[j], shuffle=400)

                    output, _ = self.mle(dataset.get_likelihood, coords, y, theta0, bounds=bounds)

                    output = torch.from_numpy(output)

                    loss = torch.abs(output - theta_test[j])

                    all_loss_test.append(loss)
                    all_theta_final.append(output)
                    all_true_theta.append(theta_test[j])

        time_total = time.time() - time_start
        full_loss_tensor = torch.stack(all_loss_test, dim=0)
        all_theta_final = torch.stack(all_theta_final, dim=0)
        all_true_theta = torch.stack(all_true_theta, dim=0)

        results = {
            "final_test_losses": full_loss_tensor,  # (Nb_epochs, Nb_params)
            "test_time": time_total,
            "train_time": None,
            "final_all_theta": all_theta_final,  # (Size_Test, Nb_params)
            "true_all_theta": all_true_theta,
        }
        self.test_time += time_total
        self._save_results(results)
