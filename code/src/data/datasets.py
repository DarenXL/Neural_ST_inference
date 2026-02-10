import torch
import numpy as np
import os
import time
from torch.utils.data import Dataset
from tqdm import tqdm
from models.base import SimulationModel
# Si tu as besoin de add_missing_mask, importe-le depuis utils
from utils import add_missing_mask


class GenericSimulationDataset(Dataset):
    def __init__(self, model: SimulationModel, theta_size=200, time_size=40, space_size=40, deepset_size=5,
                 missing_ratio=0):
        self.model = model
        self.theta_size = theta_size
        self.time_size = time_size
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
                theta_i, self.time_size, self.space_size, self.deepset_size
            )

            np.save(f"{path}/samples/sample_{idx}.npy", batch)

        self.time_generation = time.time() - start_time
        self.save(path)

    def __getitem__(self, idx):
        batch = np.load(f"{self.path}/samples/sample_{idx}.npy")

        print(self.missing_ratio)

        if self.missing_ratio > 0:
            batch = add_missing_mask(batch, self.missing_ratio)

        theta_vec = self.get_theta_tensor()[idx]

        theta_tiled = theta_vec.tile(self.deepset_size, 1)

        deepset = torch.ones(self.deepset_size, dtype=int) * idx

        return (torch.from_numpy(batch), theta_tiled, deepset)

    def get_fulldata_mask(self, idx):
        batch = np.load(f"{self.path}/samples/sample_{idx}.npy")

        if self.missing_ratio > 0:
            batch_mask = add_missing_mask(batch.copy(), self.missing_ratio)[0]
        else:
            batch_mask = batch.copy()

        batch_mask[0] = batch[0]

        theta = self.get_theta_tensor()[idx]
        return (batch_mask, theta)