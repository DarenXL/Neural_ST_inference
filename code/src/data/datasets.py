import torch
from pathlib import Path
import numpy as np
import random
import os
import time
from torch.utils.data import Dataset
from tqdm import tqdm
import concurrent.futures
from models.base import SimulationModel
from utils import CMAP, create_mask, temporary_seed
from scipy.stats import qmc
import gstlearn as gl
import matplotlib.pyplot as plt


class GenericSimulationDataset(Dataset):
    def __init__(
        self,
        model: SimulationModel,
        theta_size=200,
        time_size=30,
        space_dim=1,
        space_size=30,
        missing_ratio=0,
        seed=None,
        dev=None,
        noise=None,
    ):
        self.model = model
        self.theta_size = theta_size
        self.time_size = time_size
        self.space_dim = space_dim
        self.space_size = space_size
        self.missing_ratio = missing_ratio
        self.global_seed = seed
        self.dev = dev,
        self.noise = noise,

        self.path = None
        self.time_generation = None

        self.training = True # Shuffle mask, Shuffle Sigma and Tau



        self.thetas = self.model.sample_thetas(theta_size, seed=self.global_seed)
        self.theta_names = self.model.get_param_names(alt_format=False)
        self.theta_alt_names = self.model.get_param_names(alt_format=True)

        if not(self.dev is None):
            self.sigmas = self.sample_sigmas(theta_size)
            self.theta_names += ["dev"]
            self.theta_alt_names += [r"$\sigma$"]
            self.sigma_pos = len(self.theta_names) - 1
        if not(self.noise is None) :
            self.taus = self.sample_taus(theta_size)
            self.theta_names += ["noise"]
            self.theta_alt_names += [r"$\tau$"]
            self.tau_pos = len(self.theta_names) - 1

    def __len__(self):
        return self.theta_size

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def to_title(self, vals):
        n_drop = (1 if not (self.dev is None) else 0) + (1 if not (self.noise is None) else 0)
        res = self.model.to_title(vals[:-n_drop])
        return res + (not (self.dev is None)) * ("\n" + rf"$\sigma$: {vals[-1]:.2f}") + (not (self.noise is None)) * rf" | $\tau$: {vals[-1]:.2f}"

    def get_theta_tensor(self) -> torch.Tensor:
        """Return every theta in a tensor"""

        stack = [self.thetas[name] for name in self.model.param_names]

        res = torch.from_numpy(np.stack(stack).T)
        if not(self.dev is None) :
            res = torch.cat([res,self.sigmas], dim=1)
        if not(self.noise is None) :
            res = torch.cat([res,self.taus], dim=1)
        return res

    def get_theta_by_idx(self, idx) -> torch.Tensor:
        """Return every theta in a tensor"""
        stack = [self.thetas[name][idx] for name in self.model.param_names]

        res = torch.tensor(stack)
        if not(self.dev is None) :
            res = torch.cat([res,self.sigmas[idx]])
        if not(self.noise is None) :
            res = torch.cat([res,self.taus[idx]])
        return res

    def sample_sigmas(self, size):
        sampler = qmc.LatinHypercube(d=1, seed=self.global_seed)
        sample = sampler.random(size)

        res =  qmc.scale(sample, self.noise if not(self.noise is None) else .3, self.dev)
        return torch.from_numpy(res)

    def sample_taus(self, size):
        sampler = qmc.LatinHypercube(d=1, seed=self.global_seed)
        sample = sampler.random(size)

        res =  qmc.scale(sample, 0, self.noise)
        return torch.from_numpy(res)

    def save(self, path: Path):
        """Save all attributes and the thetas"""
        meta = {
            "config": {
                "time_size": self.time_size,
                "space_dim": self.space_dim,
                "space_size": self.space_size,
                "theta_size": self.theta_size,
                "missing_ratio": self.missing_ratio,
                "global_seed": self.global_seed,
                "dev": self.dev,
                "noise": self.noise,
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
        data = torch.load(f"{path}/metadata.pt", weights_only=False)
        config = data["config"]

        instance = cls(
            model=model_instance,
            theta_size=config["theta_size"],
            time_size=config["time_size"],
            space_dim=config["space_dim"],
            space_size=config["space_size"],
            missing_ratio=config["missing_ratio"],
            seed=config["global_seed"],
            dev=config.get("dev", 1.414),
            noise=config.get("noise", 0.5),
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
            mult_time=1,
        )
        batch = torch.from_numpy(batch)
        torch.save(batch, f"{samples_dir}/sample_{idx}.pt")
        return idx

    def generate_masks(self, path: Path):

        if self.missing_ratio is not None :
            path = path / ("ratio_"+str(self.missing_ratio))
        path = path / "masks"
        path.mkdir(parents=True, exist_ok=True)

        # somme = 0

        for idx in range(self.theta_size):
            seed_idx = None if self.global_seed is None else self.global_seed + idx
            with temporary_seed(seed_idx):
                mask = create_mask(self.space_size, self.space_dim, self.missing_ratio)
            torch.save(mask, f"{Path(path)}/mask_{idx}.pt")
            # somme += (~mask).sum()
            # plt.imshow(mask)
            # plt.show()
        # print(somme / (50*50*self.theta_size))


    def generate(self, path: Path):
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

                worker_seed = None if self.global_seed is None else self.global_seed + idx

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

        self.generate_masks(path)

    # def generate(self, path: Path, base_seed=None):
    #     """Generate gaussian fields on a complete grid"""
    #     os.makedirs(f"{path}/samples", exist_ok=True)
    #
    #     if base_seed is None:
    #         base_seed = self.global_seed
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
    #             seed = base_seed + idx
    #         )
    #
    #         torch.save(batch,f"{path}/samples/sample_{idx}.pt")
    #
    #     self.time_generation = time.time() - start_time
    #     self.save(path)
    #
    #     self.generate_masks(path)

    def generate_one(self, idx: int):
        """Generate one gaussian field on a complete grid and return it"""
        theta_i = {key: self.thetas[key][idx] for key in self.model.param_names}

        batch = self.model.simulate(
            theta_i, self.time_size, self.space_dim, self.space_size, 1
        )

        return batch

    def generate_from_theta(self, theta_values, seed=None):
        theta_dict = dict(zip(self.model.param_names, theta_values))

        batch = self.model.simulate(
            theta_dict, self.time_size, self.space_dim, self.space_size, seed
        )
        return torch.from_numpy(batch)

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
        theta = self.get_theta_by_idx(idx)

        title_text = self.model.to_title(theta)

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
                img = img.clone()
                img[~mask] = torch.nan

            ax.imshow(img, aspect="equal", extent=[0, 1, 0, 1])
            ax.set_title(
                (title_text + f"\n\nTime {time_idx}" if (mask is None) else ""),
                fontsize=10,
            )

    def sample_to_coord(self, sample, shuffle :int = None):
        y = sample[0].reshape(-1) # order="C"
        mask_obs = sample[1].clone().detach().to(torch.bool).reshape(-1)

        t = torch.linspace(0, 1, self.time_size + 1)[:-1]
        s = torch.linspace(0, 1, self.space_size + 1)[:-1]

        axes = [s, t] if self.space_dim == 1 else [s, s, t]
        grid = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack(grid, dim=-1).reshape(-1, len(axes))

        if not(shuffle is None):
            v = np.arange(coords.shape[0])[~mask_obs]
            v = v[np.random.permutation(len(v))]
            v = v[0:shuffle]

            coords = coords[v, :]
            y = y[v]
            mask_obs = mask_obs[v]

        return coords, y, mask_obs

    def kriging_from_sample(self, sample, theta):
        """Kriging"""

        coords, y, mask_obs = self.sample_to_coord(sample)
        coords = coords.detach().numpy()
        y = y.detach().numpy()
        mask_obs = mask_obs.detach().numpy()

        y_obs = y[~mask_obs]
        coords_obs = coords[~mask_obs]
        coords_miss = coords # [mask_obs]

        n_drop = (1 if not(self.dev is None) else 0) + (1 if not(self.noise is None) else 0)
        theta_dict = dict(zip(self.model.param_names, theta[:-n_drop]))

        krig = self.model.kriging(coords_in=coords_obs, coords_out=coords_miss, vals_in=y_obs, space_dim=self.space_dim, theta_values=theta_dict, dev=theta[self.sigma_pos] if not(self.dev is None) else None, noise=theta[self.tau_pos] if not(self.dev is None) else None)

        # y_full = y.copy()
        # y_full[mask_obs] = krig
        y_full = krig

        if self.space_dim == 1:
            return torch.from_numpy(y_full).reshape(
                self.space_size, self.time_size # , order="F"
            )  # , order="F"
        else:
            return torch.from_numpy(y_full).reshape(
                self.space_size, self.space_size, self.time_size # , order="F"
            )

    def get_grid(self, idx):
        """Load the sample idx"""
        res = torch.load(f"{self.path}/samples/sample_{idx}.pt")
        return res

    def get_mask(self, idx):
        if self.missing_ratio is not None :
            path = self.path / ("ratio_"+str(self.missing_ratio))
        else :
            path = self.path
        return torch.load(f"{path}/masks/mask_{idx}.pt")

    def add_missing_mask(self, sample, idx=None):

        if self.training:
            mask_idx = torch.randint(0, self.theta_size, (1,)).item()
        else:
            mask_idx = idx if idx is not None else 0

        mask = torch.ones_like(sample)
        mask[
            :,
            :,
            :,
            self.get_mask(mask_idx),
        ] = 0

        return torch.concatenate([sample, mask], dim=1)

    def add_dev(self, sample, idx=None):

        if self.training:
            sigma_idx = torch.randint(0, self.theta_size, (1,)).item()
            dev = self.sigmas[sigma_idx]
        else:
            sigma_idx = idx if idx is not None else 0
            dev = self.sigmas[sigma_idx]

        return dev * sample, self.sigmas[sigma_idx]

    def add_noise(self, sample, idx=None):

        if self.training:
            tau_idx = torch.randint(0, self.theta_size, (1,)).item()
            noise = self.taus[tau_idx] * torch.randn_like(sample)
        else:
            tau_idx = idx if idx is not None else 0
            seed = None if self.global_seed is None else self.global_seed + tau_idx

            if seed is not None:
                g = torch.Generator(device=sample.device)
                g.manual_seed(seed)
                noise = self.taus[tau_idx] * torch.randn(
                    sample.shape, generator=g, dtype=sample.dtype, device=sample.device
                )
            else:
                noise = self.taus[tau_idx] * torch.randn_like(sample)

        return  sample + noise, self.taus[tau_idx]



    def __getitem__(self, idx):

        theta_vec = self.get_theta_by_idx(idx)

        sample = self.get_grid(idx)

        if not(self.dev is None) :
            sample, sigma = self.add_dev(sample, idx)
            theta_vec[self.sigma_pos] = sigma

        if not(self.noise is None) :
            sample, tau = self.add_noise(sample, idx)
            theta_vec[self.tau_pos] = tau


        sample = self.add_missing_mask(sample, idx)

        return (sample, theta_vec.unsqueeze(0))


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

        if is_perm == 0:
            theta = self.all_thetas[sample_idx] # TODO
        else:
            theta = self.all_thetas[self.pi[sample_idx]]

        sample = self.get_grid(sample_idx)

        if not(self.dev is None) :
            sample, sigma = self.add_dev(sample, sample_idx)
            if is_perm == 0:
                theta[self.sigma_pos] = sigma
            else:
                theta[self.sigma_pos] = self.sigmas[self.pi[sample_idx]]

        if not(self.noise is None) :
            sample, tau = self.add_noise(sample, sample_idx)
            if is_perm == 0:
                theta[self.tau_pos] = tau
            else :
                theta[self.tau_pos] = self.taus[self.pi[sample_idx]]

        sample = self.add_missing_mask(sample)

        classi = torch.tensor([is_perm])

        return sample, theta.unsqueeze(0), classi
