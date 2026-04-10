import torch
from pathlib import Path
import numpy as np
import os
import time
from torch.utils.data import Dataset
from tqdm import tqdm
from models.base import SimulationModel
from utils import add_missing_mask
import gstlearn as gl
import matplotlib.pyplot as plt

class GenericSimulationDataset(Dataset):
    def __init__(
        self,
        model: SimulationModel,
        theta_size=200,
        time_size=40,
        space_dim=1,
        space_size=40,
        min_deepset_size=1,
        max_deepset_size=5,
        missing_ratio=0,
    ):
        self.model = model
        self.theta_size = theta_size
        self.time_size = time_size
        self.space_dim = space_dim
        self.space_size = space_size
        self.min_deepset_size = max(min_deepset_size, 1)
        self.max_deepset_size = max_deepset_size
        self.missing_ratio = missing_ratio

        self.path = None
        self.time_generation = None

        self.thetas = self.model.sample_thetas(theta_size)

    def __len__(self):
        return self.theta_size

    def get_theta_tensor(self) -> torch.Tensor:
        """Return every theta in a tensor"""
        stack = [self.thetas[name] for name in self.model.param_names]
        return torch.from_numpy(np.stack(stack).T)

    def save(self, path: Path):
        """Save all attributes and the thetas"""
        meta = {
            "config": {
                "time_size": self.time_size,
                "space_dim": self.space_dim,
                "space_size": self.space_size,
                "theta_size": self.theta_size,
                "min_deepset_size": self.min_deepset_size,
                "max_deepset_size": self.max_deepset_size,
                "missing_ratio": self.missing_ratio,
                "time_generation": self.time_generation,
                "model_class": self.model.__class__.__name__,
            },
            "thetas": self.thetas,
        }
        np.save(f"{path}/metadata.npy", meta)
        self.path = path

    @classmethod
    def load(cls, path: Path, model_instance: SimulationModel):
        """Load all attributes and the thetas"""
        data = np.load(f"{path}/metadata.npy", allow_pickle=True).item()
        config = data["config"]

        instance = cls(
            model=model_instance,
            theta_size=config["theta_size"],
            time_size=config["time_size"],
            space_dim=config["space_dim"],
            space_size=config["space_size"],
            min_deepset_size=config["min_deepset_size"],
            max_deepset_size=config["max_deepset_size"],
            missing_ratio=config["missing_ratio"],
        )

        instance.thetas = data["thetas"]
        instance.time_generation = config["time_generation"]
        instance.path = path

        return instance

    def generate(self, path: Path):
        """Generate gaussian fields on a complete grid"""
        os.makedirs(f"{path}/samples", exist_ok=True)

        t_iter = tqdm(range(self.theta_size), total=self.theta_size)
        start_time = time.time()

        for idx in t_iter:
            theta_i = {key: self.thetas[key][idx] for key in self.model.param_names}

            batch = self.model.simulate(
                theta_i,
                self.time_size,
                self.space_dim,
                self.space_size,
                self.max_deepset_size,
            )

            np.save(f"{path}/samples/sample_{idx}.npy", batch)

        self.time_generation = time.time() - start_time
        self.save(path)

    def generate_one(self, idx: int) -> np.ndarray:
        """Generate one gaussian field on a complete grid and return it"""
        theta_i = {key: self.thetas[key][idx] for key in self.model.param_names}

        batch = self.model.simulate(
            theta_i, self.time_size, self.space_dim, self.space_size, 1
        )

        return batch

    def generate_from_theta(self, theta_values: np.ndarray) -> np.ndarray:
        theta_dict = dict(zip(self.model.param_names, theta_values))

        batch = self.model.simulate(
            theta_dict, self.time_size, self.space_dim, self.space_size, 1
        )
        return batch

    def get_min_max(self, k:list[int]):
        all_values = []
        for ki in k:
            batch, _ = self.get_grid(ki, deepset_size=self.max_deepset_size)
            all_values.append(batch[:, 0])

        combined_data = np.stack(all_values)

        v_min = float(np.min(combined_data))
        v_max = float(np.max(combined_data))

        return v_min, v_max

    def show(self, ax, idx: int, mask: np.ndarray = None, sample_idx: int = 0, time_idx: int = 0):
        """Show a gaussian field"""
        batch, _ = self.get_grid(idx, deepset_size=1, shuffle=False)
        theta = self.get_theta_tensor()[idx]

        title_text = self.model.to_title(theta, alt_format=True)

        if self.space_dim == 1:

            img = batch[sample_idx, 0]

            if not (mask is None):
                img = img.copy()
                img[:,~mask] = np.nan

            ax.imshow(img, aspect="equal", extent=[0, 1, 0, 1])
            ax.set_title(title_text  if (mask is None) else "", fontsize=10)

        elif self.space_dim == 2:

            img = batch[sample_idx, 0, time_idx]

            if not (mask is None):
                img = img.copy()
                img[~mask] = np.nan

            ax.imshow(img, aspect="equal", extent=[0, 1, 0, 1])
            ax.set_title(
                (
                    title_text + f"\n\nTime {time_idx}" if (mask is None) else ""
                ),
                fontsize=10,
            )

    def sample_to_coord(self, sample):
        y, mask_obs = sample[0].reshape(-1), np.array(sample[1], dtype=bool).reshape(
            -1
        )  # .T.reshape(-1,order="F")

        t = np.linspace(0, 1, self.time_size)
        s = np.linspace(0, 1, self.space_size)

        axes = [t, s] if self.space_dim == 1 else [t, s, s]
        grid = np.meshgrid(*axes, indexing="ij")
        coords = np.stack(grid, axis=-1).reshape(-1, len(axes))

        # print(np.sum(mask_obs))

        return coords, y, mask_obs

    def kriging_from_sample(self, sample: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Kriging"""

        coords, y, mask_obs = self.sample_to_coord(sample)

        y_obs = y[~mask_obs]
        coords_obs = coords[~mask_obs]
        coords_miss = coords[mask_obs]

        krig = self.model.kriging(
            coords_obs, coords_miss, y_obs, self.space_dim, theta
        )

        y_full = y.copy()
        y_full[mask_obs] = krig

        if self.space_dim == 1:
            return y_full.reshape(self.time_size, self.space_size)  # , order="F"
        else:
            return y_full.reshape(self.time_size, self.space_size, self.space_size)

    def get_likelihood_from_sample(self, sample: np.ndarray, theta: np.ndarray):

        coords, y, mask_obs = self.sample_to_coord(sample)

        y_obs = y[~mask_obs]
        coords_obs = coords[~mask_obs]
        # coords_miss = coords[mask_obs]

        # print("get_likelihood_from_sample", coords_obs.shape)

        return self.model.get_likelihood(coords_obs, y_obs, theta)

    def get_likelihood(self, coords_obs: np.ndarray, y_obs: np.ndarray, theta: np.ndarray):

        return self.model.get_likelihood(coords_obs, y_obs, theta)

    def get_grid(self, idx, deepset_size=None, shuffle=True):
        """Load the sample idx"""
        batch = np.load(f"{self.path}/samples/sample_{idx}.npy")

        if deepset_size is None :
            deepset_size = torch.randint(
                low=self.min_deepset_size, high=self.max_deepset_size + 1, size=(1,)
            ).item()
        deepset_idx = torch.randperm(self.max_deepset_size)[:deepset_size]

        if shuffle :
            batch = batch[deepset_idx]
            if deepset_size == 1:
                batch = np.expand_dims(batch, axis=0)
        else :
            batch = batch[np.arange(deepset_size)]

        return batch, deepset_idx

    def __getitem__(self, idx):

        batch, deepset_idx = self.get_grid(idx)
        deepset_size = len(deepset_idx)

        batch = add_missing_mask(batch, self.missing_ratio)

        theta_vec = self.get_theta_tensor()[idx]

        theta_tiled = theta_vec.tile(deepset_size, 1)

        deepset = torch.ones(deepset_size, dtype=int) * idx

        return (torch.from_numpy(batch), theta_tiled, deepset)


class NlsSimulationDataset(GenericSimulationDataset):
    def __init__(self, dataset: GenericSimulationDataset, simu_nls=False):
        self.__dict__ = dataset.__dict__.copy()

        self.all_thetas = self.get_theta_tensor()
        self.simu_nls = simu_nls

        self.pi = torch.stack(
            [
                torch.randint(0, self.theta_size, (self.theta_size,))
                for _ in range(self.max_deepset_size)
            ]
        )

    def __len__(self):
        return 2 * self.theta_size

    def get_data_theta_classi(self, idx):
        is_perm = idx // self.theta_size
        sample_idx = idx % self.theta_size

        batch, deepset_idx = self.get_grid(sample_idx, self.max_deepset_size)
        deepset_size = len(deepset_idx)

        batch = add_missing_mask(batch, self.missing_ratio)

        if is_perm == 0:
            true_theta = self.all_thetas[sample_idx]
            theta = true_theta.repeat(deepset_size, 1)
        else:
            fake_indices = self.pi[:, sample_idx]
            theta = self.all_thetas[fake_indices][:deepset_size]

        classi = torch.full((deepset_size,), is_perm, dtype=torch.int64)

        return torch.from_numpy(batch), theta, classi

    def __getitem__(self, idx):

        # print(batch.shape)

        if not self.simu_nls :
            return self.get_data_theta_classi(idx)
        else :

            is_perm = idx // self.theta_size
            sample_idx = idx % self.theta_size

            full_batch = np.load(f"{self.path}/samples/sample_{sample_idx}.npy")
            deepset_size = torch.randint(
                low=self.min_deepset_size, high=self.max_deepset_size, size=(1,)
            ).item()
            deepset_idx = torch.randperm(self.max_deepset_size)
            batch = full_batch[deepset_idx[:deepset_size]]

            if deepset_size == 1:
                batch = np.expand_dims(batch, axis=0)

            # print(batch.shape, deepset_size, deepset_idx)

            batch = add_missing_mask(batch, self.missing_ratio)

            if is_perm == 0:

                # theta = full_batch[deepset_idx[deepset_size:]]
                # if deepset_size == self.max_deepset_size-1:
                #     theta = np.expand_dims(theta, axis=0)
                # print(is_perm, theta.shape)
                # theta = np.expand_dims(theta, axis=0)
                # theta = torch.from_numpy(np.tile(theta, (deepset_size,1,1,1,1)))

                theta = np.concatenate(
                        [
                            np.load(
                                f"{self.path}/samples/sample_{sample_idx}.npy"
                            )[deepset_idx[deepset_size:]]
                            for i in range(deepset_size)
                        ]
                    )

                # print(is_perm, theta.shape)
            else:

                theta = np.concatenate(
                        [
                            np.load(
                                f"{self.path}/samples/sample_{self.pi[0,sample_idx]}.npy"
                            )[deepset_idx[deepset_size:]]
                            for i in range(deepset_size)
                        ]
                    )
                # print(is_perm, theta.shape)

            theta = torch.from_numpy(theta)
            if len(theta.shape) == 3 :
                theta = theta.unsqueeze(dim=1)
            theta = torch.cat((theta, torch.ones_like(theta)),dim=1)
            # print(theta.shape)

            # print(torch.ones(deepset_size * (self.max_deepset_size - deepset_size), dtype=int) * sample_idx,
            #     torch.arange(deepset_size).repeat_interleave(self.max_deepset_size - deepset_size))

            deepset = torch.stack(
                (torch.ones(deepset_size * (self.max_deepset_size - deepset_size), dtype=int) * idx, # sample_idx,
                torch.arange(deepset_size).repeat_interleave(self.max_deepset_size - deepset_size)),
            ).T
            # print(deepset)

            classi = torch.full((deepset_size,), is_perm, dtype=torch.int64)

            # print()

            return torch.from_numpy(batch), theta, deepset, classi
