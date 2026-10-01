"""
Gneiting Spatio-Temporal Covariance Kernel
==========================================

    C(h,u) = phi(u)^{-beta*d/2} * K_nu( ||h|| / (a_s * phi(u)^{beta/2}) )
    phi(u) = 1 + |u/a_t|^alpha,   alpha in (0,2],  beta in (0,1]

Trace kernels: Gauss, Matern (nu in {0.5,1.5,2.5} for KeOps; any nu for dense),
               Cauchy (nu>0, learnable).

Architecture:
  GneitingKernelDense   -- full n x m matrix, any Matern nu.
  GneitingKernelKeOps   -- O(N) memory via GneitingLinearOperator backed by KeOps.
  GneitingKernel        -- factory (picks KeOps when available).

NOTE on representation_tree compatibility
-----------------------------------------
linear_operator.LinearOperator reconstructs objects via
    cls(*tensor_args, **nondifferentiable_kwargs)
where tensor_args come from representation() (the *args to super().__init__)
and nondifferentiable_kwargs are the non-tensor **kwargs passed to super().__init__.

GneitingLinearOperator follows this contract by passing d, trace_type, nu_fixed
as **kwargs to super().__init__() — LinearOperator handles the splitting.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive, Interval

try:
    import pykeops  # noqa
    from pykeops.torch import LazyTensor as KEOps_LT
    KEOPS_AVAILABLE = True
except ImportError:
    KEOPS_AVAILABLE = False
    import warnings
    warnings.warn("pykeops not found. Using dense fallback.")

# ---------------------------------------------------------------------------
# Dense spatial trace kernels
# ---------------------------------------------------------------------------

class SpatialTraceKernel(nn.Module):
    def forward(self, r: Tensor) -> Tensor:
        raise NotImplementedError


class GaussTraceKernel(SpatialTraceKernel):
    def forward(self, r: Tensor) -> Tensor:
        return torch.exp(-(r ** 2))


class MaternTraceKernel(SpatialTraceKernel):
    def __init__(self, nu: float = 1.5, learn_nu: bool = False):
        super().__init__()
        if learn_nu:
            self.register_parameter("raw_nu", nn.Parameter(torch.tensor(math.log(nu))))
            self._nu_constraint = Positive()
        else:
            self.register_buffer("fixed_nu", torch.tensor(nu))
            self.raw_nu = None
        self.learn_nu = learn_nu

    @property
    def nu(self) -> Tensor:
        return self._nu_constraint.transform(self.raw_nu) if self.learn_nu else self.fixed_nu

    def forward(self, r: Tensor) -> Tensor:
        nu = self.nu
        eps = 1e-10
        r = r.clamp(min=eps)
        log_coeff = (1.0 - nu) * math.log(2) - torch.lgamma(nu)
        sqrt2nu_r = (2.0 * nu).sqrt() * r
        sqrt2nu_r = sqrt2nu_r.clamp(min=eps)
        if hasattr(torch.special, "modified_bessel_second_kind"):
            log_kv = torch.log(
                torch.special.modified_bessel_second_kind(nu, sqrt2nu_r).clamp(min=1e-38)
            )
        else:
            import scipy.special as sc
            kv = torch.tensor(sc.kv(nu.item(), sqrt2nu_r.detach().cpu().numpy()),
                              dtype=r.dtype, device=r.device)
            log_kv = torch.log(kv.clamp(min=1e-38))
        return (log_coeff + nu * torch.log(sqrt2nu_r) + log_kv).exp()


class CauchyTraceKernel(SpatialTraceKernel):
    def __init__(self, nu: float = 1.0, learn_nu: bool = True):
        super().__init__()
        self.register_parameter("raw_nu", nn.Parameter(torch.tensor(math.log(nu))))
        self._nu_constraint = Positive()

    @property
    def nu(self) -> Tensor:
        return self._nu_constraint.transform(self.raw_nu)

    def forward(self, r: Tensor) -> Tensor:
        return (1.0 + r ** 2) ** (-self.nu)


# ---------------------------------------------------------------------------
# Dense Gneiting kernel
# ---------------------------------------------------------------------------

class GneitingKernelDense(Kernel):
    """
    Gneiting spatio-temporal kernel — dense (materialises full n x m matrix).
    Supports any Matern nu (including learned). d=2 + 1 time by default.
    Input: x of shape (..., n, d+1), last column = time.
    """
    has_lengthscale = False

    def __init__(self, spatial_dim=2, trace_kernel="matern", nu=1.5,
                 learn_nu=False, adv_init=0,
                 ls_s_init=1.0, ls_t_init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d = spatial_dim
        tk = trace_kernel.lower()
        if tk == "gauss":
            self.trace = GaussTraceKernel()
        elif tk == "matern":
            self.trace = MaternTraceKernel(nu=nu, learn_nu=learn_nu)
        elif tk == "cauchy":
            self.trace = CauchyTraceKernel(nu=nu, learn_nu=learn_nu)
        else:
            raise ValueError(f"Unknown trace kernel: {tk!r}")

        self.register_parameter("raw_adv", nn.Parameter(torch.tensor(math.log(adv_init))))
        self._adv_constraint = Interval(-1.0, 1.0)
        self.register_parameter("raw_ls_s", nn.Parameter(torch.tensor(math.log(ls_s_init))))
        self.register_parameter("raw_ls_t", nn.Parameter(torch.tensor(math.log(ls_t_init))))
        self._ls_constraint = Positive()

    @property
    def adv(self): return self._adv_constraint.transform(self.raw_adv)
    @property
    def length_scale_s(self): return self._ls_constraint.transform(self.raw_ls_s)
    @property
    def length_scale_t(self): return self._ls_constraint.transform(self.raw_ls_t)

    def forward(self, x1, x2, **params):
        s1, t1 = x1[..., : self.d], x1[..., self.d :]
        s2, t2 = x2[..., : self.d], x2[..., self.d :]

        # 1. Lag temporel AVEC son signe pour l'advection
        dt = t1.unsqueeze(-2) - t2.unsqueeze(-3)  # Shape (..., N, M, 1)

        # 2. Facteur temporel phi (utilise la valeur absolue pour la stationnarité temporelle)
        phi = 1.0 + (dt.abs().squeeze(-1) / self.length_scale_t + 1e-8) ** 2
        tf = phi ** (-0.5 * self.beta * self.d)

        # 3. Application de l'advection spatiale : s_diff - adv * dt
        # self.adv est un scalaire (ou vecteur). dt a une forme (..., N, M, 1)
        s_diff = (s1.unsqueeze(-2) - s2.unsqueeze(-3)) - self.adv * dt
        ds2 = torch.sum(s_diff**2, dim=-1)  # Distance spatiale toujours >= 0

        # 4. Calcul de la distance normalisée r (plus de soustraction instable ici)
        r = (ds2 + 1e-20).sqrt() / (self.length_scale_s * phi ** (0.5 * 1))
        return tf * self.trace(r)


# ---------------------------------------------------------------------------
# KeOps-backed LinearOperator + KeOps kernel
# ---------------------------------------------------------------------------

if KEOPS_AVAILABLE:
    from linear_operator import LinearOperator as _LOBase
    from gpytorch.kernels.keops.keops_kernel import KeOpsKernel

    class GneitingLinearOperator(_LOBase):
        """
        Custom LinearOperator for the Gneiting kernel powered by KeOps MVM.

        WHY NOT KernelLinearOperator
        ----------------------------
        KernelLinearOperator._diagonal() and ._matmul() call to_dense() on
        the covar_func return value, which fails for KeOps LazyTensors.

        representation_tree CONTRACT
        ----------------------------
        linear_operator rebuilds this object via:
            cls(*tensor_args, **nondifferentiable_kwargs)
        where tensor_args = self.representation() = (x1, x2, alpha, beta,
        ls_s, ls_t [, nu_tensor]) and nondifferentiable_kwargs = the non-tensor
        kwargs passed to super().__init__: d, trace_type, nu_fixed.

        We must therefore:
          1. Pass tensors as positional *args to super().__init__().
          2. Pass non-tensor scalars as **kwargs to super().__init__() —
             LinearOperator separates them automatically into differentiable
             (tensor) and non-differentiable (scalar) buckets.

        _matmul  : KeOps MVM — O(N+M) memory.
        _diagonal: k(xi,xi)=1 for any stationary kernel at zero lag.
        """

        def __init__(self, x1, x2, adv, ls_s, ls_t,
                     d=2, trace_type="matern", nu_fixed=1.5, nu_tensor=None):
            # ---------------------------------------------------------------
            # Tensors go as positional *args; non-tensor scalars go as **kwargs.
            # LinearOperator.__init__ splits kwargs automatically:
            #   - Tensor kwargs  → _differentiable_kwargs  (tracked in representation)
            #   - Non-tensor kwargs → non-differentiable kwargs stored in the
            #     RepresentationTree and re-passed as **kwargs on reconstruction.
            # ---------------------------------------------------------------
            if nu_tensor is not None:
                super().__init__(x1, x2, adv, ls_s, ls_t, nu_tensor,
                                 d=d, trace_type=trace_type, nu_fixed=nu_fixed)
            else:
                super().__init__(x1, x2, adv, ls_s, ls_t,
                                 d=d, trace_type=trace_type, nu_fixed=nu_fixed)

            # Convenience attributes for use in _matmul, _diagonal, etc.
            self.x1_         = x1
            self.x2_         = x2
            self.adv_        = adv
            self.ls_s_       = ls_s
            self.ls_t_       = ls_t
            self.d_          = d
            self.trace_type_ = trace_type
            self.nu_fixed_   = float(nu_fixed)
            self.nu_tensor_  = nu_tensor

        def _size(self):
            return torch.Size([*self.x1_.shape[:-2],
                               self.x1_.shape[-2], self.x2_.shape[-2]])

        def _transpose_nonbatch(self):
            return self.__class__(
                self.x2_, self.x1_,
                self.adv_, self.ls_s_, self.ls_t_,
                d=self.d_, trace_type=self.trace_type_,
                nu_fixed=self.nu_fixed_, nu_tensor=self.nu_tensor_,
            )

        def _build_keops_K(self, x1, x2):
            """
            Build KeOps LazyTensor K(xi, xj) — shape (..., N, M, 1).
            Parameters extracted via .item(): acceptable because autograd
            flows through the dense output of _matmul, not through the KeOps
            symbolic graph.
            """
            d  = self.d_
            # trainable scalar parameters kept in graph
            v_  = KEOps_LT(self.adv_.reshape(1, 1, 1))
            ls_ = KEOps_LT(self.ls_s_.reshape(1, 1, 1))
            lt_ = KEOps_LT(self.ls_t_.reshape(1, 1, 1))

            # Split before converting to LazyTensor
            s1 = x1[..., :d]          # (..., N, d)
            t1 = x1[..., d:d+1]       # (..., N, 1)

            s2 = x2[..., :d]          # (..., M, d)
            t2 = x2[..., d:d+1]       # (..., M, 1)

            # Convert separately to KeOps LazyTensors
            si = KEOps_LT(s1.contiguous().unsqueeze(-2))  # (..., N, 1, d)
            sj = KEOps_LT(s2.contiguous().unsqueeze(-3))  # (..., 1, M, d)

            ti = KEOps_LT(t1.contiguous().unsqueeze(-2))  # (..., N, 1, 1)
            tj = KEOps_LT(t2.contiguous().unsqueeze(-3))  # (..., 1, M, 1)

            # 1. Lag temporel signé pour l'advection
            dt = ti - tj

            # 2. phi avec lag absolu
            du = dt.abs() / lt_
            phi = 1.0 + (du + 1e-8) ** 2
            tf = phi ** (-0.5 * d * 1)

            # 3. Distance spatiale modifiée par l'advection
            s_diff = (si - sj) - v_ * dt
            r2 = (s_diff**2).sum(-1) / (ls_**2 * phi**1)

            tt = self.trace_type_
            if tt == "gauss":
                k = (-r2).exp()

            elif tt == "cauchy":
                nv_ = (
                    KEOps_LT(self.nu_tensor_.reshape(1, 1, 1))
                    if self.nu_tensor_ is not None
                    else self.nu_fixed_
                )
                k = (1.0 + r2) ** (-nv_)

            else:  # matern — half-integer
                nu = self.nu_fixed_
                r = (r2 + 1e-20).sqrt()  # Strictement positif, plus de - v_ * tf ici !
                s = math.sqrt(2.0 * nu) * r
                if abs(nu - 0.5) < 1e-3:
                    k = (-s).exp()
                elif abs(nu - 1.5) < 1e-3:
                    k = (1.0 + s) * (-s).exp()
                else:  # nu = 2.5
                    k = (1.0 + s + s**2 / 3.0) * (-s).exp()

            return tf * k  # (N, M, 1) KeOps LazyTensor

        def _matmul(self, rhs: Tensor) -> Tensor:
            """
            K @ rhs — fully vectorised KeOps reduction, O(N+M) memory.

            rhs : (M, P)  →  out : (N, P)

            All P probe vectors are handled in one KeOps kernel call:
              K  (N, M, 1)  *  rhs_j (1, M, P)  →  sum_j  →  (N, P)
            This is P× faster than a column loop and avoids P JIT compilations.
            """
            K     = self._build_keops_K(self.x1_, self.x2_)  # (N, M, 1)
            rhs_j = KEOps_LT(rhs.contiguous().unsqueeze(0))   # (1, M, P)
            return (K * rhs_j).sum(dim=1)                     # (N, P)

        def _get_indices(self, row_index, col_index, *batch_indices):
            """
            Compute K[row_index[k], col_index[k]] for each k — pointwise.

            WHY THIS IS CRITICAL FOR MEMORY
            --------------------------------
            The default LinearOperator._get_indices calls to_dense(), which
            materialises the full N×N kernel matrix (9000² × 4 = 324 MB).
            pivoted_cholesky (the CG preconditioner) calls _get_indices ~100
            times. Each call creates a 324 MB tensor with a grad_fn that holds
            a reference to the full matrix — totalling ~32 GB retained on GPU.

            This override evaluates the kernel pointwise in O(N) memory.
            .item() is safe here: _get_indices is only called for the
            preconditioner, which does not need gradients w.r.t. parameters.
            """
            x1_sub = self.x1_[..., row_index, :]   # (..., P, d+1)
            x2_sub = self.x2_[..., col_index, :]   # (..., P, d+1)
            d  = self.d_
            v  = self.adv_.item()
            ls = self.ls_s_.item()
            lt = self.ls_t_.item()

            s1, t1 = x1_sub[..., :d], x1_sub[..., d:]  # (..., P, d), (..., P, 1)
            s2, t2 = x2_sub[..., :d], x2_sub[..., d:]

            # 1. Lag temporel signé
            dt = t1 - t2
            phi = 1.0 + (dt.abs() / lt + 1e-8) ** 2  # (..., P, 1)
            tf = phi ** (-0.5 * 1 * d)

            # 2. Distance spatiale avec advection pointwise
            s_diff = (s1 - s2) - v * dt
            ds2 = (s_diff**2).sum(-1, keepdim=True)  # (..., P, 1)
            r2 = ds2 / (ls**2 * phi**1)

            tt = self.trace_type_
            if tt == "gauss":
                k = (-r2).exp()
            elif tt == "cauchy":
                nv = (
                    self.nu_tensor_.item()
                    if self.nu_tensor_ is not None
                    else self.nu_fixed_
                )
                k = (1.0 + r2) ** (-nv)
            else:  # matern half-integer
                nu = self.nu_fixed_
                r = (r2 + 1e-20).sqrt()
                s = math.sqrt(2.0 * nu) * r
                if abs(nu - 0.5) < 1e-3:
                    k = (-s).exp()
                elif abs(nu - 1.5) < 1e-3:
                    k = (1.0 + s) * (-s).exp()
                else:
                    k = (1.0 + s + s**2 / 3.0) * (-s).exp()

            return (tf * k).squeeze(-1)  # (..., P)

        def _diagonal(self) -> Tensor:
            """
            k(xi, xi): for any stationary kernel at zero lag, phi=1, r=0 => k=1.
            Returns ones of shape (..., N).
            """
            n = self.x1_.shape[-2]
            return torch.ones(*self.x1_.shape[:-2], n,
                              dtype=self.x1_.dtype, device=self.x1_.device)

    class GneitingKernelKeOps(KeOpsKernel):
        """
        Gneiting spatio-temporal kernel — KeOps implementation.
        forward() returns a GneitingLinearOperator (KeOps-backed, O(N) memory).
        Supported: trace_kernel in {'gauss','matern','cauchy'}.
                   Matern: nu in {0.5, 1.5, 2.5}.
        Input: x of shape (..., n, d+1), last column = time.
        """
        has_lengthscale = False

        def __init__(self, spatial_dim=2, trace_kernel="matern", nu=1.5,
                     learn_nu=False, adv_init=0.,
                     ls_s_init=1.0, ls_t_init=1.0, **kwargs):
            super().__init__(**kwargs)
            self.d = spatial_dim
            tk = trace_kernel.lower()
            if tk not in ("gauss", "matern", "cauchy"):
                raise ValueError(f"Unknown trace kernel: {tk!r}")
            if tk == "matern" and nu not in (0.5, 1.5, 2.5):
                raise ValueError(
                    f"KeOps Matern requires nu in {{0.5,1.5,2.5}}, got {nu}."
                    " Use GneitingKernelDense for arbitrary nu."
                )
            self.trace_type = tk
            self.nu_fixed   = float(nu)
            self._adv_constraint = Interval(-1.0, 1.0)

            # self.register_parameter("raw_alpha", nn.Parameter(torch.tensor(math.log(alpha_init))))
            # self.register_parameter("raw_beta",  nn.Parameter(torch.tensor(math.log(beta_init))))
            raw_adv = self._adv_constraint.inverse_transform(torch.tensor(adv_init))

            self.register_parameter("raw_adv", nn.Parameter(raw_adv))

            self.register_parameter("raw_ls_s", nn.Parameter(torch.tensor(math.log(ls_s_init))))
            self.register_parameter("raw_ls_t", nn.Parameter(torch.tensor(math.log(ls_t_init))))
            self._ls_constraint = Positive()
            if tk == "cauchy":
                self.register_parameter("raw_nu", nn.Parameter(torch.tensor(math.log(nu))))
                self._nu_constraint = Positive()
            else:
                self.raw_nu = None

        @property
        def adv(self):
            return self._adv_constraint.transform(self.raw_adv)

        @property
        def length_scale_s(self):
            return self._ls_constraint.transform(self.raw_ls_s)

        @property
        def length_scale_t(self):
            return self._ls_constraint.transform(self.raw_ls_t)

        @property
        def nu(self):
            return self._nu_constraint.transform(self.raw_nu) if self.raw_nu is not None else None

        def forward(self, x1, x2, diag=False, **params):
            nu_tensor = self.nu if self.trace_type == "cauchy" else None
            res = GneitingLinearOperator(
                x1, x2,
                self.adv, self.length_scale_s, self.length_scale_t,
                d=self.d, trace_type=self.trace_type,
                nu_fixed=self.nu_fixed, nu_tensor=nu_tensor,
            )
            return res.diagonal(dim1=-1, dim2=-2) if diag else res


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def GneitingKernel(spatial_dim=2, trace_kernel="matern", nu=1.5,
                   learn_nu=False, adv_init=0.0,
                   ls_s_init=1.0, ls_t_init=1.0, use_keops=None, **kwargs):
    """
    Factory: returns GneitingKernelKeOps if KeOps available, else GneitingKernelDense.

    Input: x of shape (..., n, spatial_dim+1), last column = time.
    Wrap with gpytorch.kernels.ScaleKernel for variance sigma^2.

    Args:
        spatial_dim:  d (default 2 for 2D space + 1 time).
        trace_kernel: 'gauss', 'matern', or 'cauchy'.
        nu:           Smoothness (Matern ∈ {0.5,1.5,2.5} for KeOps; any for dense).
        learn_nu:     Learn nu by gradient (dense only).
        alpha_init:   Initial alpha in (0,2].
        beta_init:    Initial beta  in (0,1].
        ls_s_init:    Initial spatial  length-scale a_s.
        ls_t_init:    Initial temporal length-scale a_t.
        use_keops:    True / False / None (auto).
    """
    kw = dict(spatial_dim=spatial_dim, trace_kernel=trace_kernel, nu=nu,
              learn_nu=learn_nu, adv_init=adv_init,
              ls_s_init=ls_s_init, ls_t_init=ls_t_init, **kwargs)
    if use_keops is True or (use_keops is None and KEOPS_AVAILABLE):
        try:
            return GneitingKernelKeOps(**kw)
        except (ImportError, ValueError):
            pass
    return GneitingKernelDense(**kw)


# ---------------------------------------------------------------------------
# GP model
# ---------------------------------------------------------------------------

class GneitingGP(gpytorch.models.ExactGP):
    """
    Exact GP with Gneiting spatio-temporal kernel.

    train_x : (n, spatial_dim+1) — last column is time.
    train_y : (n,)
    """

    def __init__(self, train_x, train_y, likelihood, spatial_dim=2,
                 trace_kernel="matern", nu=1.5, learn_nu=False,
                 adv_init=0.0, ls_s_init=1.0, ls_t_init=1.0,
                 use_keops=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(GneitingKernel(
            spatial_dim=spatial_dim, trace_kernel=trace_kernel, nu=nu,
            learn_nu=learn_nu, adv_init=adv_init,
            ls_s_init=ls_s_init, ls_t_init=ls_t_init, use_keops=use_keops,
        ))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ---------------------------------------------------------------------------
# Training utility
# ---------------------------------------------------------------------------

def train_gneiting_gp(train_x, train_y, spatial_dim=2, trace_kernel="matern",
                      nu=1.5, learn_nu=False, n_iter=200, lr=0.05,
                      tau_init=0.1,
                      use_keops=None, verbose=True):
    """
    Train a GneitingGP by marginal log-likelihood (Adam optimiser).
    """
    device = train_x.device

    # 1. La Likelihood contient le paramètre de bruit (variance de l'erreur de mesure)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Initialisation de tau (le paramètre 'noise' de GPyTorch correspond à tau^2)
    likelihood.noise = tau_init ** 2

    model = GneitingGP(train_x, train_y, likelihood, spatial_dim=spatial_dim,
                       trace_kernel=trace_kernel, nu=nu, learn_nu=learn_nu,
                       use_keops=use_keops).to(device)

    model.train()
    likelihood.train()

    # L'optimiseur mettra à jour à la fois les paramètres du modèle (adv, a_s, a_t)
    # et de la vraisemblance (tau)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []

    # Calcul du facteur de conversion une seule fois pour l'affichage
    sq12 = math.sqrt(12)

    for i in range(n_iter):
        optimizer.zero_grad()

        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and (i + 1) % 20 == 0:
            k = model.covar_module.base_kernel

            # Récupération de tau = sqrt(noise)
            tau_est = torch.sqrt(likelihood.noise).item()

            print(f"Iter {i+1:4d}/{n_iter} | Loss {loss.item():.4f} | "
                  f"adv={k.adv.item():.3f} "
                  f"a_t={k.length_scale_t.item():.3f}  "
                  f"a_s={k.length_scale_s.item()/sq12:.3f}  "
                  f"tau={tau_est:.3f}"
                )

    model.eval()
    likelihood.eval()
    return model, likelihood, losses


# ---------------------------------------------------------------------------
# Self-test (dense path only, no pykeops required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    d, n = 2, 300                              # 2D space + 1 time  (d+1 = 3 cols)
    X = torch.randn(n, d + 1)
    y = torch.sin(X[:, 0]) * torch.cos(X[:, 1]) + 0.1 * torch.randn(n)
    print("=== GneitingGP self-test  (dense, Matern 3/2, d=2) ===")
    model, lik, loss = train_gneiting_gp(X, y, spatial_dim=d, trace_kernel="matern",
                                   nu=1.5, n_iter=60, lr=0.05, use_keops=False)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(model(torch.randn(5, d + 1)))

    print("Predictive means:", pred.mean.numpy().round(3))
    print("Self-test passed.")
