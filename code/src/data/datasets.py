import torch
from pathlib import Path
import numpy as np
import os
import time
from torch.utils.data import Dataset
from tqdm import tqdm
import concurrent.futures
from models.base import SimulationModel
from utils import add_missing_mask, CMAP, create_mask
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
        missing_ratio=0,
    ):
        self.model = model
        self.theta_size = theta_size
        self.time_size = time_size
        self.space_dim = space_dim
        self.space_size = space_size
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
                "missing_ratio": self.missing_ratio,
                "time_generation": self.time_generation,
                "model_class": self.model.__class__.__name__,
            },
            "thetas": self.thetas,
        }
        torch.save(meta, f"{path}/metadata.pt")
        self.path = path

    @classmethod
    def load(cls, path: Path, model_instance: SimulationModel):
        """Load all attributes and the thetas"""
        data = torch.load(f"{path}/metadata.pt")
        config = data["config"]

        instance = cls(
            model=model_instance,
            theta_size=config["theta_size"],
            time_size=config["time_size"],
            space_dim=config["space_dim"],
            space_size=config["space_size"],
            missing_ratio=config["missing_ratio"],
        )

        instance.thetas = data["thetas"]
        instance.time_generation = config["time_generation"]
        instance.path = path

        return instance

    @staticmethod
    def _worker_simulate_and_save(
        idx,
        theta_i,
        model_instance,
        time_size,
        space_dim,
        space_size,
        samples_dir,
        seed,
    ):
        batch = model_instance.simulate(
            theta_i,
            time_size,
            space_dim,
            space_size,
            seed,
        )
        batch = torch.from_numpy(batch)
        torch.save(batch, f"{samples_dir}/sample_{idx}.pt")
        return idx

    def generate(self, path: Path, base_seed: int = 42):
        samples_dir = Path(path) / "samples"
        os.makedirs(samples_dir, exist_ok=True)

        start_time = time.time()

        max_workers = None

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = []
            for idx in range(self.theta_size):
                theta_i = {key: self.thetas[key][idx] for key in self.model.param_names}

                worker_seed = base_seed + idx

                f = executor.submit(
                    GenericSimulationDataset._worker_simulate_and_save,
                    idx,
                    theta_i,
                    self.model,
                    self.time_size,
                    self.space_dim,
                    self.space_size,
                    samples_dir,
                    worker_seed,
                )
                futures.append(f)

            t_iter = tqdm(
                concurrent.futures.as_completed(futures),
                total=self.theta_size,
                desc="Simulations",
            )

            for future in t_iter:
                _ = future.result()
        self.time_generation = time.time() - start_time
        self.save(path)

        for idx in range(self.theta_size):
            mask = create_mask(self.space_size, self.space_dim, self.missing_ratio)
            torch.save(mask, f"{Path(path) / "masks"}/mask_{idx}.pt")

    # def generate(self, path: Path):
    #     """Generate gaussian fields on a complete grid"""
    #     os.makedirs(f"{path}/samples", exist_ok=True)
    #
    #     t_iter = tqdm(range(self.theta_size), total=self.theta_size)
    #     start_time = time.time()
    #
    #     for idx in t_iter:
    #         theta_i = {key: self.thetas[key][idx] for key in self.model.param_names}
    #
    #         batch = self.model.simulate(
    #             theta_i,
    #             self.time_size,
    #             self.space_dim,
    #             self.space_size,
    #         )
    #
    #         np.save(f"{path}/samples/sample_{idx}.npy", batch)
    #
    #     self.time_generation = time.time() - start_time
    #     self.save(path)

    def generate_one(self, idx: int):
        """Generate one gaussian field on a complete grid and return it"""
        theta_i = {key: self.thetas[key][idx] for key in self.model.param_names}

        batch = self.model.simulate(
            theta_i, self.time_size, self.space_dim, self.space_size, 1
        )

        return batch

    def generate_from_theta(self, theta_values):
        theta_dict = dict(zip(self.model.param_names, theta_values))

        batch = self.model.simulate(
            theta_dict, self.time_size, self.space_dim, self.space_size, 1
        )
        return batch

    def get_min_max(self, k: list[int]):
        all_values = []
        for ki in k:
            batch = self.get_grid(ki)
            all_values.append(batch[:, 0])

        combined_data = torch.stack(all_values)

        v_min = float(torch.min(combined_data))
        v_max = float(torch.max(combined_data))

        return v_min, v_max

    def show(
        self,
        ax,
        idx: int,
        mask = None,
        sample_idx: int = 0,
        time_idx: int = 0,
    ):
        """Show a gaussian field"""
        batch = self.get_grid(idx)
        theta = self.get_theta_tensor()[idx]

        title_text = self.model.to_title(theta, alt_format=True)

        if self.space_dim == 1:

            img = batch[sample_idx, 0]

            if not (mask is None):
                img = img.clone()
                img[:, ~mask] = torch.nan

            ax.imshow(img, aspect="equal", extent=[0, 1, 0, 1], cmap=CMAP)
            ax.set_title(title_text if (mask is None) else "", fontsize=10)

        elif self.space_dim == 2:

            img = batch[sample_idx, 0, time_idx]

            if not (mask is None):
                img = img.copy()
                img[~mask] = torch.nan

            ax.imshow(img, aspect="equal", extent=[0, 1, 0, 1])
            ax.set_title(
                (title_text + f"\n\nTime {time_idx}" if (mask is None) else ""),
                fontsize=10,
            )

    def sample_to_coord(self, sample):
        y = sample[0].reshape(-1, order="C")
        mask_obs = torch.tensor(sample[1], dtype=torch.bool).T.reshape(-1)

        t = torch.linspace(0, 1, self.time_size + 1)[:-1]
        s = torch.linspace(0, 1, self.space_size + 1)[:-1]

        axes = [s, t] if self.space_dim == 1 else [s, s, t]
        grid = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack(grid, dim=-1).reshape(-1, len(axes))

        return coords, y, mask_obs

    def kriging_from_sample(self, sample, theta):
        """Kriging"""

        coords, y, mask_obs = self.sample_to_coord(sample)

        y_obs = y[~mask_obs]
        coords_obs = coords[~mask_obs]
        coords_miss = coords[mask_obs]

        krig = self.model.kriging(coords_obs, coords_miss, y_obs, self.space_dim, theta)

        y_full = y.copy()
        y_full[mask_obs] = krig

        if self.space_dim == 1:
            return y_full.reshape(
                self.space_size, self.time_size, order="F"
            )  # , order="F"
        else:
            return y_full.reshape(
                self.space_size, self.space_size, self.time_size, order="F"
            )

    def get_grid(self, idx):
        """Load the sample idx"""
        return torch.load(f"{self.path}/samples/sample_{idx}.pt")

    def get_mask(self, idx):
        return torch.load(f"{self.path}/masks/mask_{idx}.pt")

    def add_missing_mask(self, batch, idx=None):

        if idx is None :
            idx = torch.randint(0, self.theta_size, (1,))

        mask = torch.ones_like(batch)

        mask[
            :,
            :,
            :,
            self.get_mask(idx),
        ] = 0

        return torch.concatenate([batch, mask], dim=1)

    def __getitem__(self, idx):

        batch = self.get_grid(idx)

        batch = add_missing_mask(batch, self.missing_ratio)

        theta_vec = self.get_theta_tensor()[idx]

        return (batch, theta_vec.unsqueeze(0))


class NlsSimulationDataset(GenericSimulationDataset):
    def __init__(self, dataset: GenericSimulationDataset):
        self.__dict__ = dataset.__dict__.copy()

        self.all_thetas = self.get_theta_tensor()

        self.pi = torch.randint(0, self.theta_size, (self.theta_size,))

    def __len__(self):
        return 2 * self.theta_size

    def __getitem__(self, idx):

        is_perm = idx // self.theta_size
        sample_idx = idx % self.theta_size

        batch = self.get_grid(sample_idx)

        batch = add_missing_mask(batch, self.missing_ratio)

        if is_perm == 0:
            theta = self.all_thetas[sample_idx]
        else:
            theta = self.all_thetas[self.pi[sample_idx]]

        classi = torch.tensor([is_perm])

        return batch, theta.unsqueeze(0), classi
