from abc import ABC
from pathlib import Path

import torch
import time
from tqdm import tqdm

from neural_st_inference.models.utils_gp import evaluate_gneiting_kriging


class BaseEstimator(ABC):
    """Base class for NBE, NLE and MLE for training, testing and plot graphs"""

    def __init__(self, device, res_path, estimation_label: str, nls: bool = False):
        self.device = device
        self.res_path = Path(res_path)
        self.estimation_label = estimation_label
        self.nls = nls

        self.theta_names = None
        self.theta_alt_names = None

        self.simu_time = 0
        self.train_time = 0
        self.test_time = 0

        self.train_dataset_hist = []
        self.test_dataset_hist = []

        (self.res_path / "complete_outputs").mkdir(parents=True, exist_ok=True)

    def _save_results(self, outputs, prefix="outputs"):
        idx = len(self.test_dataset_hist) - 1
        filename = f"{prefix}_{self.estimation_label} [{idx}].pt"
        path = self.res_path / "complete_outputs" / filename
        torch.save(outputs, path)

    def pred_error(self, theta_pred, dataset_idx=-1, n_train=300, n_test=1000):
        """
        WARNING : Adapted to Gneiting matèrn model only
        """
        dataset = self.test_dataset_hist[dataset_idx]
        all_pred_scores = []

        time_start = time.time()
        sample_idx = 0

        for i in tqdm(range(len(dataset)), desc=f"Pred Error {self.estimation_label}"):
            data_test, _ = dataset[i]
            data_test = data_test.detach().float()

            for j in range(len(data_test)):
                coords_full, y_full, mask_obs = dataset.sample_to_coord(
                    data_test[j], shuffle=None
                )

                coords_full = coords_full.to(self.device, dtype=torch.float32)
                y_full = y_full.to(self.device, dtype=torch.float32)
                mask_obs = mask_obs.to(self.device, dtype=torch.bool)

                obs_indices = torch.nonzero(~mask_obs, as_tuple=False).squeeze(-1)
                perm = torch.randperm(obs_indices.size(0), device=self.device)

                actual_n_train = min(n_train, obs_indices.size(0))
                train_idx = obs_indices[perm[:actual_n_train]]
                train_x, train_y = coords_full[train_idx], y_full[train_idx]

                masked_indices = torch.nonzero(mask_obs, as_tuple=False).squeeze(-1)
                if masked_indices.numel() > 0:
                    test_idx = masked_indices[:n_test]
                else:
                    test_idx = obs_indices[
                        perm[actual_n_train : actual_n_train + n_test]
                    ]

                test_x, test_y = coords_full[test_idx], y_full[test_idx]

                current_theta = (
                    theta_pred[i, j] if theta_pred.ndim == 3 else theta_pred[sample_idx]
                )
                current_theta = current_theta.to(self.device, dtype=torch.float32)

                mse, mae, nlpd = evaluate_gneiting_kriging(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    current_theta,
                    dataset,
                    self.device,
                )

                all_pred_scores.append(torch.tensor([mse, mae, nlpd]))
                sample_idx += 1

        results = {
            "prediction_scores": torch.stack(all_pred_scores, dim=0),
            "test_time": time.time() - time_start,
        }
        self._save_results(results, prefix="pred_scores")
        return results
