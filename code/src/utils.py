from pathlib import Path
from datetime import datetime
import numpy as np
import torch

theta_colors = [
    {
        "train":"orangered",
        "test":"lightsalmon",
        "baseline":"dimgrey"
    },
    {
        "train":"deepskyblue",
        "test":"lightskyblue",
        "baseline":"dimgrey"
    },
    {
        "train":"forestgreen",
        "test":"limegreen",
        "baseline":"dimgrey"
    },
    {
        "train":"darkorchid",
        "test":"orchid",
        "baseline":"dimgrey"
    },
]

theta_names_adv = ["advection", "range time", "range space", "std"]
theta_names_sep = ["separability", "scale time", "scale space", "sill"]

def create_data_path(base_path: str) -> Path:
    path = Path(base_path).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = path / timestamp

    sub_dirs = [
        path / "train" / "samples",
        path / "test" / "samples"
    ]

    for sd in sub_dirs:
        sd.mkdir(parents=True, exist_ok=True)

    return path

def create_res_path(base_path: str) -> Path:
    path = Path(base_path).resolve()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = path / timestamp

    sub_dirs = [
        path / "loss_evolution",
        path / "conclusion_graphs",
        path / "models"
    ]

    for sd in sub_dirs:
        sd.mkdir(parents=True, exist_ok=True)

    return path

def add_missing_mask(batch, missing_ratio):
    space_size = batch.shape[-1]
    nb_missing = int(space_size * missing_ratio)
    nb_visible = space_size - nb_missing

    batch = batch.copy()

    if nb_visible <= (space_size + 1) // 2:
        max_index = space_size - (nb_visible - 1)
        base = np.random.choice(max_index, size=nb_visible, replace=False)
        base = np.sort(base)
        visibles = base + np.arange(nb_visible)
    else:
        visibles = np.random.choice(space_size, size=nb_visible, replace=False)

    mask = np.ones_like(batch)
    mask[:, :, :, visibles] = 0

    batch[:, :, :, mask[0, 0, 0] == 1] = 0

    return np.concatenate([batch, mask], axis=1)

def deepset_collate(batch):
    data, theta, deepset = map(torch.stack, zip(*batch))

    data = data.flatten(0, 1)
    theta = theta.flatten(0, 1)
    deepset = deepset.flatten(0, 1)

    return data, theta, deepset


def unique_preserve_order(x):
    seen = []
    out = []
    for v in x.tolist():
        if v not in seen:
            seen.append(v)
            out.append(v)
    return torch.tensor(out, device=x.device)


def map_deepset(deepset, theta, device):
    unique_ds = unique_preserve_order(deepset)

    deepset_mapped = torch.tensor(
        [unique_ds.tolist().index(int(ds)) for ds in deepset],
        device=device
    )

    theta = torch.stack([theta[deepset == v][0] for v in unique_ds])

    return deepset_mapped, theta


def baseline_mae(dataset):
    all_thetas = dataset.get_theta()

    theta_mean = all_thetas.mean(dim=0)

    preds = theta_mean.unsqueeze(0).repeat(len(dataset), 1)

    mae = torch.mean(torch.abs(preds - all_thetas), dim=0)

    return mae, theta_mean

def global_mean_pool(x, deepset):
    num_groups = deepset.max().item() + 1
    out = torch.zeros(num_groups, x.size(1), device=x.device)
    out.index_add_(0, deepset, x)
    return out / torch.bincount(deepset).float().unsqueeze(1)