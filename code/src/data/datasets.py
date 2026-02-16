import torch
import numpy as np
import os
import time
from torch.utils.data import Dataset
from tqdm import tqdm
from models.base import SimulationModel
# Si tu as besoin de add_missing_mask, importe-le depuis utils
from utils import add_missing_mask
import matplotlib.pyplot as plt


class GenericSimulationDataset(Dataset):
    def __init__(self, model: SimulationModel, theta_size=200, time_size=40, space_dim=1, space_size=40, deepset_size=5,
                 missing_ratio=0):
        self.model = model
        self.theta_size = theta_size
        self.time_size = time_size
        self.space_dim = space_dim
        self.space_size = space_size
        self.deepset_size = deepset_size
        self.missing_ratio = missing_ratio

        self.path = None
        self.time_generation = None

        self.thetas = self.model.sample_thetas(theta_size)

    def __len__(self):
        return self.theta_size

    def get_theta_tensor(self):
        stack = [self.thetas[name] for name in self.model.param_names]
        return torch.from_numpy(np.stack(stack).T)

    def save(self, path):
        meta = {
            "config": {
                "time_size": self.time_size,
                "space_dim": self.space_dim,
                "space_size": self.space_size,
                "theta_size": self.theta_size,
                "deepset_size": self.deepset_size,
                "missing_ratio": self.missing_ratio,
                "time_generation": self.time_generation,
                "model_class": self.model.__class__.__name__
            },
            "thetas": self.thetas
        }
        np.save(f"{path}/metadata.npy", meta)
        self.path = path

    @classmethod
    def load(cls, path, model_instance: SimulationModel):
        data = np.load(f"{path}/metadata.npy", allow_pickle=True).item()
        config = data["config"]

        instance = cls(
            model=model_instance,
            theta_size=config["theta_size"],
            time_size=config["time_size"],
            space_dim=config["space_dim"],
            space_size=config["space_size"],
            deepset_size=config["deepset_size"],
            missing_ratio=config["missing_ratio"]
        )

        instance.thetas = data["thetas"]
        instance.time_generation = config["time_generation"]
        instance.path = path

        return instance

    def generate(self, path):
        os.makedirs(f"{path}/samples", exist_ok=True)

        t_iter = tqdm(range(self.theta_size), total=self.theta_size)
        start_time = time.time()

        for idx in t_iter:
            theta_i = {key: self.thetas[key][idx] for key in self.model.param_names}

            batch = self.model.simulate(
                theta_i, self.time_size, self.space_dim, self.space_size, self.deepset_size
            )

            np.save(f"{path}/samples/sample_{idx}.npy", batch)

        self.time_generation = time.time() - start_time
        self.save(path)

    def generate_one(self, idx):

        theta_i = {key: self.thetas[key][idx] for key in self.model.param_names}

        batch = self.model.simulate(
            theta_i, self.time_size, self.space_dim, self.space_size, 1
        )

        return batch

    def show(self, ax, idx, sample_idx=0):
        batch, theta, _ = self.__getitem__(idx)

        if self.space_dim == 1:

            img = batch[sample_idx, 0]

            title_text = self.model.to_title(theta[0])

            ax.imshow(img, aspect='equal', extent=[0,1,0,1])
            # ax.colorbar(label="Signal")
            ax.set_title(title_text, fontsize=10)
            # ax.set_xlabel("Space")
            # ax.set_ylabel("Time")

        elif self.space_dim == 2:

            for i in range(len(ax)):

                img = batch[sample_idx, 0, i]

                title_text = self.model.to_title(theta[0])

                ax[i].imshow(img, aspect='equal', extent=[0, 1, 0, 1])
                ax[i].set_title(f"Time {i}" if i != len(ax)//2 else title_text+f"\n\nTime {i}", fontsize=10)
                # ax[i].set_xlabel("Space 1")
                # ax[i].set_ylabel(r"$\it{Space 2}$")

    def kriging(self, sample, theta_pred):
        y, mask_obs = sample[0].reshape(-1), np.array(sample[1], dtype=bool).reshape(-1)

        t = np.linspace(0, 1, self.time_size)
        s = np.linspace(0, 1, self.space_size)

        if self.space_dim == 1 :
            coords = np.array([(ti, si) for ti in t for si in s])
        else :
            coords = np.array([(ti, si, sj) for ti in t for si in s for sj in s])

        y_obs = y[~mask_obs]
        coords_obs = coords[~mask_obs]
        coords_miss = coords[mask_obs]

        cov_oo = self.model.get_cov(
            coords_obs, coords_obs, self.space_dim, theta_pred
        )

        cov_mo = self.model.get_cov(
            coords_miss, coords_obs, self.space_dim, theta_pred
        )

        cov_oo += 1e-6 * np.eye(cov_oo.shape[0])

        y_miss = cov_mo @ np.linalg.solve(cov_oo, y_obs)

        y_full = y.copy()
        y_full[mask_obs] = y_miss

        if self.space_dim == 1 :
            return y_full.reshape(self.time_size, self.space_size)
        else :
            return y_full.reshape(self.time_size, self.space_size, self.space_size)

    def __getitem__(self, idx):
        batch = np.load(f"{self.path}/samples/sample_{idx}.npy")

        if self.missing_ratio > 0:
            batch = add_missing_mask(batch, self.missing_ratio)

        theta_vec = self.get_theta_tensor()[idx]

        theta_tiled = theta_vec.tile(self.deepset_size, 1)

        deepset = torch.ones(self.deepset_size, dtype=int) * idx

        return (torch.from_numpy(batch), theta_tiled, deepset)


class NlsSimulationDataset(GenericSimulationDataset):
    def __init__(self, dataset: GenericSimulationDataset):
        self.__dict__ = dataset.__dict__.copy()

        self.all_thetas = self.get_theta_tensor()

        self.pi = torch.stack([
            torch.randint(0, self.theta_size, (self.theta_size,))
            for _ in range(self.deepset_size)
        ])

    def __len__(self):
        return 2 * self.theta_size

    def __getitem__(self, idx):
        is_perm = idx // self.theta_size
        sample_idx = idx % self.theta_size

        batch = np.load(f"{self.path}/samples/sample_{sample_idx}.npy")
        if self.missing_ratio > 0:
            batch = add_missing_mask(batch, self.missing_ratio)

        if is_perm == 0:
            true_theta = self.all_thetas[sample_idx]
            theta = true_theta.repeat(self.deepset_size, 1)
        else:
            fake_indices = self.pi[:, sample_idx]
            theta = self.all_thetas[fake_indices]

        classi = torch.full((self.deepset_size,), is_perm, dtype=torch.int64)

        return torch.from_numpy(batch), theta, classi
