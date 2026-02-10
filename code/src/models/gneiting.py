import time
from scipy.special import gamma, kv
import numpy as np
from .base import SimulationModel


def variogram(u, alpha=1.0, b=1.0):
    """Variogram: gamma(u) = (|u|/b)^alpha"""
    return (np.abs(u) / b) ** alpha


def matern_cov(h, nu=1.5, a=1.0):
    """Matern isotropic covariance function"""
    d = np.abs(h)
    res = np.ones_like(d)
    mask = d > 0

    if np.any(mask):
        d_val = d[mask]
        const = 2.0 / gamma(nu)
        term = (d_val / (2.0 * a)) ** nu
        bessel = kv(nu, d_val / a)
        res[mask] = const * term * bessel

    return res


def sim_w(p, Nt, params):
    """
    Simulates p copies of temporal processes W with temporal variogram
    gamma*(u) = (gamma(u)+1)^beta - 1
    """
    # Params: [nu, a, alpha, b, beta, delta, k]
    alpha, b_param, beta = params[2], params[3], params[4]

    t_indices = np.arange(1, Nt + 1)

    c1 = (variogram(-t_indices, alpha=alpha, b=b_param) + 1.0) ** beta - 1.0

    diff_matrix = t_indices[:, None] - t_indices[None, :]
    cc_term = (variogram(diff_matrix, alpha=alpha, b=b_param) + 1.0) ** beta - 1.0

    cc = c1[:, None] + c1[None, :] - cc_term

    L = np.linalg.cholesky(cc + 1e-10 * np.eye(Nt))

    X = np.random.normal(size=(Nt, p))

    W = L @ X
    return W.T


def sim_z1(p, Nt, params):
    """
    Simulates p copies of temporal processes Z1 with temporal covariance function
    gamma*(u) = (gamma(u)+1)^(-delta)
    """
    alpha, b_param, delta = params[2], params[3], params[5]

    t_indices = np.arange(1, Nt + 1)
    diff_matrix = t_indices[:, None] - t_indices[None, :]
    cc = 1.0 / (variogram(diff_matrix, alpha=alpha, b=b_param) + 1.0) ** delta

    L = np.linalg.cholesky(cc + 1e-10 * np.eye(Nt))
    X = np.random.normal(size=(Nt, p))

    Z = L @ X
    return Z.T  # (p, Nt)


def sim_rvs(p, params):
    """
    Simulates all necessary Random Variables for the
    spectral band simulation algorithm with substitution approach
    """
    nu, a = params[0], params[1]
    k = params[6]

    r_gamma = np.random.gamma(shape=nu, scale=2 * (a ** 2), size=p)

    R_val = 1.0 / (2.0 * r_gamma)
    multR = np.sqrt(2.0 * R_val)

    Phi = 2 * np.pi * np.random.uniform(size=p)

    U_rand = np.random.uniform(size=p)
    UU = np.sqrt(-2.0 * np.log(U_rand))

    Omega = np.random.normal(size=(k, p))

    return {
        "UU": UU,  # (p,)
        "multR": multR,  # (p,)
        "Omega": Omega,  # (k, p)
        "Phi": Phi  # (p,)
    }


def ztilde_t_vecs(s, t_idx, rvs, W, Z1, sigma):
    """
    Do the computation of the simulated RF
    s: coordinates (Ns, k)
    t_idx: time (0 to Nt-1)
    """
    UU = rvs["UU"]
    multR = rvs["multR"]
    Omega = rvs["Omega"]
    Phi = rvs["Phi"]

    nOmega = np.linalg.norm(Omega, axis=0)  # (p,)

    ww_t = W[:, t_idx]  # (p,)
    zz_t = Z1[:, t_idx]  # (p,)

    p = len(UU)

    spatial_proj = s @ Omega
    term_cos = (spatial_proj * multR) + (nOmega * ww_t / np.sqrt(2)) + Phi
    V = np.cos(term_cos)

    V = V * UU * zz_t

    field_t = np.sum(V, axis=1) / np.sqrt(p)

    return field_t * sigma


def simulate_spatio_temporal(params_dict, Nt, s_coords, k=1):
    """Eq (14), Allard et al. 2025"""
    p = 3000

    # les noms dans geomodel
    nu = params_dict['smooth_s']
    a = params_dict['scale_s']
    alpha = params_dict['power_t']
    b = params_dict['scale_t']
    bet = params_dict['sep']

    delta = params_dict['power2_t'] - bet * k / 2  # sep = tau | Allard et al. 2025

    sim_params = [nu, a, alpha, b, bet, delta, k]

    sigma = np.sqrt(params_dict['sill'])

    W = sim_w(p, Nt, sim_params)  # (p, Nt)
    Z1 = sim_z1(p, Nt, sim_params)  # (p, Nt)
    RVs = sim_rvs(p, sim_params)

    Ns = s_coords.shape[0]
    Z_field = np.zeros((Ns, Nt))

    for t in range(Nt):
        Z_field[:, t] = ztilde_t_vecs(s_coords, t, RVs, W, Z1, sigma)

    return Z_field

class GneitingModel(SimulationModel):
    def __init__(self):
        self._names = ['sep', 'power2_t', 'scale_time', 'scale_space', 'sill']
        self.lower_bound = (0.01, None, 1., 1., .1)
        self.upper_bound = (1., None, 5., 5., 1.)

    @property
    def param_names(self):
        return self._names

    def sample_thetas(self, n):
        sep = np.random.uniform(self.lower_bound[0], self.upper_bound[0], (n,))
        power2_t = np.random.uniform(sep, np.ones_like(sep) * 3, (n,))

        return {
            'sep': sep,
            'power2_t': power2_t,
            'scale_time': np.random.uniform(self.lower_bound[2], self.upper_bound[2], (n,)),
            'scale_space': np.random.uniform(self.lower_bound[3], self.upper_bound[3], (n,)),
            'sill': np.random.uniform(self.lower_bound[4], self.upper_bound[4], (n,)) ** 2
        }

    def simulate(self, p, t_size, s_size, d_size):
        s_coords = np.expand_dims(np.arange(1, s_size + 1), 1)

        sim_params = {
            'mean': 0,
            'sill': p['sill'],
            'smooth_s': 1.,
            'scale_s': p['scale_space'],
            'power_t': 1.,
            'scale_t': p['scale_time'],
            'sep': p['sep'],
            'power2_t': p['power2_t']
        }

        batch = []
        for _ in range(d_size):
            res = simulate_spatio_temporal(sim_params, t_size, s_coords).T
            batch.append(res)

        return np.expand_dims(np.stack(batch), 1)