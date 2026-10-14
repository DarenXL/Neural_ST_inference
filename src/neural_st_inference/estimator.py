from abc import ABC
from pathlib import Path
import torch

import math
import torch
import torch.nn as nn
from torch import Tensor

import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive, Interval

import pykeops
from pykeops.torch import LazyTensor as KEOps_LT
from linear_operator import LinearOperator as _LOBase
from gpytorch.kernels.keops.keops_kernel import KeOpsKernel


# ---------------------------------------------------------------------------
# KeOps-backed LinearOperator
# ---------------------------------------------------------------------------

class GneitingLinearOperator(_LOBase):
    """
    Custom LinearOperator for the Gneiting kernel powered by KeOps MVM.
    Strictly restricted to Matern covariance.
    """

    def __init__(self, x1, x2, adv, ls_s, ls_t, d=2, nu_fixed=1.5):
        super().__init__(x1, x2, adv, ls_s, ls_t, d=d, nu_fixed=nu_fixed)

        self.x1_ = x1
        self.x2_ = x2
        self.adv_ = adv
        self.ls_s_ = ls_s
        self.ls_t_ = ls_t
        self.d_ = d
        self.nu_fixed_ = float(nu_fixed)

    def _size(self):
        return torch.Size([*self.x1_.shape[:-2], self.x1_.shape[-2], self.x2_.shape[-2]])

    def _transpose_nonbatch(self):
        return self.__class__(
            self.x2_, self.x1_,
            self.adv_, self.ls_s_, self.ls_t_,
            d=self.d_, nu_fixed=self.nu_fixed_,
        )

    def _build_keops_K(self, x1, x2):
        d = self.d_

        # Trainable scalar parameters kept in graph
        v_ = KEOps_LT(self.adv_.reshape(1, 1, 1))
        ls_ = KEOps_LT(self.ls_s_.reshape(1, 1, 1))
        lt_ = KEOps_LT(self.ls_t_.reshape(1, 1, 1))

        # Split before converting to LazyTensor
        s1, t1 = x1[..., :d], x1[..., d:d + 1]
        s2, t2 = x2[..., :d], x2[..., d:d + 1]

        si = KEOps_LT(s1.contiguous().unsqueeze(-2))
        sj = KEOps_LT(s2.contiguous().unsqueeze(-3))
        ti = KEOps_LT(t1.contiguous().unsqueeze(-2))
        tj = KEOps_LT(t2.contiguous().unsqueeze(-3))

        # 1. Lag temporel signé pour l'advection
        dt = ti - tj

        # 2. Facteur temporel phi
        du = dt.abs() / lt_
        phi = 1.0 + (du + 1e-8) ** 2
        tf = phi ** (-0.5 * d * 1)

        # 3. Distance spatiale modifiée par l'advection
        s_diff = (si - sj) - v_ * dt
        r2 = (s_diff ** 2).sum(-1) / (ls_ ** 2 * phi)

        # 4. Matern Kernel
        nu = self.nu_fixed_
        r = (r2 + 1e-20).sqrt()
        s = math.sqrt(2.0 * nu) * r

        if abs(nu - 0.5) < 1e-3:
            k = (-s).exp()
        elif abs(nu - 1.5) < 1e-3:
            k = (1.0 + s) * (-s).exp()
        else:  # nu = 2.5
            k = (1.0 + s + s ** 2 / 3.0) * (-s).exp()

        return tf * k

    def _matmul(self, rhs: Tensor) -> Tensor:
        K = self._build_keops_K(self.x1_, self.x2_)
        rhs_j = KEOps_LT(rhs.contiguous().unsqueeze(0))
        return (K * rhs_j).sum(dim=1)

    def _get_indices(self, row_index, col_index, *batch_indices):
        x1_sub = self.x1_[..., row_index, :]
        x2_sub = self.x2_[..., col_index, :]
        d = self.d_

        v = self.adv_.item()
        ls = self.ls_s_.item()
        lt = self.ls_t_.item()

        s1, t1 = x1_sub[..., :d], x1_sub[..., d:]
        s2, t2 = x2_sub[..., :d], x2_sub[..., d:]

        dt = t1 - t2
        phi = 1.0 + (dt.abs() / lt + 1e-8) ** 2
        tf = phi ** (-0.5 * d)

        s_diff = (s1 - s2) - v * dt
        ds2 = (s_diff ** 2).sum(-1, keepdim=True)
        r2 = ds2 / (ls ** 2 * phi)

        nu = self.nu_fixed_
        r = (r2 + 1e-20).sqrt()
        s = math.sqrt(2.0 * nu) * r

        if abs(nu - 0.5) < 1e-3:
            k = (-s).exp()
        elif abs(nu - 1.5) < 1e-3:
            k = (1.0 + s) * (-s).exp()
        else:
            k = (1.0 + s + s ** 2 / 3.0) * (-s).exp()

        return (tf * k).squeeze(-1)

    def _diagonal(self) -> Tensor:
        n = self.x1_.shape[-2]
        return torch.ones(*self.x1_.shape[:-2], n, dtype=self.x1_.dtype, device=self.x1_.device)


# ---------------------------------------------------------------------------
# KeOps Kernel Module
# ---------------------------------------------------------------------------

class GneitingKernel(KeOpsKernel):
    """
    Gneiting spatio-temporal kernel (Matern only) — KeOps implementation.
    """
    has_lengthscale = False

    def __init__(self, spatial_dim=2, nu=1.5, adv_init=0., ls_s_init=1.0, ls_t_init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d = spatial_dim
        self.nu_fixed = float(nu)

        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"KeOps Matern requires nu in {{0.5,1.5,2.5}}, got {nu}.")

        self._adv_constraint = Interval(-1.0, 1.0)
        raw_adv = self._adv_constraint.inverse_transform(torch.tensor(adv_init))

        self.register_parameter("raw_adv", nn.Parameter(raw_adv))
        self.register_parameter("raw_ls_s", nn.Parameter(torch.tensor(math.log(ls_s_init))))
        self.register_parameter("raw_ls_t", nn.Parameter(torch.tensor(math.log(ls_t_init))))
        self._ls_constraint = Positive()

    @property
    def adv(self): return self._adv_constraint.transform(self.raw_adv)

    @property
    def length_scale_s(self): return self._ls_constraint.transform(self.raw_ls_s)

    @property
    def length_scale_t(self): return self._ls_constraint.transform(self.raw_ls_t)

    def forward(self, x1, x2, diag=False, **params):
        res = GneitingLinearOperator(
            x1, x2, self.adv, self.length_scale_s, self.length_scale_t,
            d=self.d, nu_fixed=self.nu_fixed
        )
        return res.diagonal(dim1=-1, dim2=-2) if diag else res


# ---------------------------------------------------------------------------
# GP model
# ---------------------------------------------------------------------------

class GneitingGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, spatial_dim=2, nu=1.5,
                 adv_init=0.0, ls_s_init=1.0, ls_t_init=1.0):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Le GneitingKernel instancié ici est désormais directement la version KeOps
        self.covar_module = gpytorch.kernels.ScaleKernel(GneitingKernel(
            spatial_dim=spatial_dim, nu=nu, adv_init=adv_init,
            ls_s_init=ls_s_init, ls_t_init=ls_t_init
        ))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ---------------------------------------------------------------------------
# Training & Utility
# ---------------------------------------------------------------------------

def set_gp_params_from_theta(model, likelihood, theta, sq12_convention=True):
    covar_module = model.covar_module
    k = covar_module.base_kernel

    adv_val, at_val, as_val, sigma_val, tau_val = theta

    if sq12_convention:
        as_val = as_val * math.sqrt(12)

    with torch.no_grad():
        k.raw_adv.copy_(k._adv_constraint.inverse_transform(torch.as_tensor(adv_val, device=k.raw_adv.device)))
        k.raw_ls_t.copy_(k._ls_constraint.inverse_transform(torch.as_tensor(at_val, device=k.raw_ls_t.device)))
        k.raw_ls_s.copy_(k._ls_constraint.inverse_transform(torch.as_tensor(as_val, device=k.raw_ls_s.device)))

        covar_module.outputscale = torch.clamp(torch.tensor(sigma_val ** 2), min=1e-6)
        likelihood.noise = torch.clamp(torch.tensor(tau_val ** 2), min=1e-6)


def train_gneiting_gp(train_x, train_y, spatial_dim=2, nu=1.5,
                      n_iter=200, lr=0.05, tau_init=0.1, sigma2_init=1.0):
    device = train_x.device

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = tau_init ** 2

    model = GneitingGP(train_x, train_y, likelihood, spatial_dim=spatial_dim, nu=nu).to(device)
    model.covar_module.outputscale = sigma2_init

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []

    for i in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        # Utilisation de .detach() pour ne pas bloquer le GPU avec les transferts
        losses.append(loss.detach())

    losses = [l.item() for l in losses]

    model.eval()
    likelihood.eval()

    return model, likelihood, losses



class BaseEstimator(ABC):
    """Classe de base fournissant l'interface commune pour l'inférence et les graphiques."""

    def __init__(self, device, res_path, estimation_label: str, nls: bool = False):
        self.device = device
        self.res_path = Path(res_path)
        self.estimation_label = estimation_label
        self.nls = nls

        # Noms des paramètres
        self.theta_names = None
        self.theta_alt_names = None

        # Métriques temporelles
        self.simu_time = 0
        self.train_time = 0
        self.test_time = 0

        # Historiques de jeux de données
        self.train_dataset_hist = []
        self.test_dataset_hist = []

        # Dossier commun de sortie
        (self.res_path / "complete_outputs").mkdir(parents=True, exist_ok=True)

    def _save_results(self, outputs, prefix="outputs"):
        """Sauvegarde générique basée sur `estimation_label`."""
        idx = len(self.test_dataset_hist) - 1
        filename = f"{prefix}_{self.estimation_label} [{idx}].pt"
        path = self.res_path / "complete_outputs" / filename
        torch.save(outputs, path)