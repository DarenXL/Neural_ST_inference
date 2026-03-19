import numpy as np
import gstlearn as gl
from .base import SimulationModel
from scipy.special import gamma, kv
from scipy.stats import qmc

def _simulate_matern_grid_func(time_size, space_dim, space_size, adv, range_time, range_space, std, deepset_size=5):
    adv = np.round(adv).astype(int)
    range_time = np.maximum(range_time, np.array(0.05))
    range_space = np.maximum(range_space, np.array(0.05))
    std = np.maximum(std, np.array(0.1))

    if space_dim == 1:
        model = gl.Model.createFromParam(gl.ECov.MATERN, ranges=[range_time, range_space], param=1, sill=std ** 2,
                                         flagRange=False)

        shape_adv = abs(adv) + 1

        grid = gl.DbGrid.create(
            nx=[time_size, space_size + time_size * shape_adv],
            dx=[1. / time_size, 1. / space_size]
        )

        mesh = gl.MeshETurbo(grid)

        pop = gl.PrecisionOpMatrix(mesh, model.getCovAniso(0))

        batch = []

        for i in range(deepset_size):

            res = pop.simulateOne()
            res = res.reshape((time_size, space_size + time_size * shape_adv), order='F')

            t, s = np.meshgrid(np.arange(time_size), np.arange(space_size), indexing='ij')
            offset = t.copy() * abs(adv)
            s = s + offset

            if adv >= 0:
                res = res[t, s]
            else:
                res = np.fliplr(res)
                res = res[t, s]
                res = np.fliplr(res)

            batch.append(res)

    elif space_dim == 2 :
        sp = gl.SpaceRN.create(3)
        model = gl.Model.createFromParam(gl.ECov.MATERN, ranges=[range_time, range_space, range_space], param=1, sill=std ** 2,
                                         flagRange=False, space = sp)

        shape_adv = abs(adv) + 1

        grid = gl.DbGrid.create(
            nx=[time_size, space_size + time_size * shape_adv, space_size],
            dx=[1. / time_size, 1. / space_size, 1. / space_size]
        )

        mesh = gl.MeshETurbo(grid)

        pop = gl.PrecisionOpMatrix(mesh, model.getCovAniso(0))

        batch = []

        for i in range(deepset_size):

            res = pop.simulateOne()
            res = res.reshape((time_size, space_size + time_size * shape_adv, space_size), order='F')

            t, s1, s2 = np.meshgrid(np.arange(time_size), np.arange(space_size), np.arange(space_size), indexing='ij')
            offset = t.copy() * abs(adv)
            s1 = s1 + offset

            if adv >= 0:
                res = res[t, s1, s2]
            else:
                res = np.fliplr(res)
                res = res[t, s1, s2]
                res = np.fliplr(res)

            batch.append(res)
        # else:
        #     raise "space_dim not 1 or 2"

    return np.expand_dims(np.stack(batch), 1)


def matern_adv_cov(coords1, coords2, space_dim, adv, range_time, range_space, std):

    range_time = max(range_time, 0.05)
    range_space = max(range_space, 0.05)
    std = max(std, 0.1)

    if space_dim == 1:
        t1 = coords1[:, 0].reshape(-1, 1)
        s1 = coords1[:, 1].reshape(-1, 1)

        t2 = coords2[:, 0].reshape(1, -1)
        s2 = coords2[:, 1].reshape(1, -1)

        adv = int(round(adv))

        # Advection
        s1 = s1 + adv * t1
        s2 = s2 + adv * t2

        nu = 1.0

        d2 = ((t1 - t2) ** 2) / range_time**2 + ((s1 - s2) ** 2) / range_space**2
        d = np.sqrt(d2 + 1e-12)

        K = (
            std**2
            * (2 ** (1 - nu))
            / gamma(nu)
            * (np.sqrt(2 * nu) * d) ** nu
            * kv(nu, np.sqrt(2 * nu) * d)
        )

        return K
    elif space_dim == 2:
        t1 = coords1[:, 0].reshape(-1, 1)
        s1 = coords1[:, 1].reshape(-1, 1)
        s1_ = coords1[:, 2].reshape(-1, 1)

        t2 = coords2[:, 0].reshape(1, -1)
        s2 = coords2[:, 1].reshape(1, -1)
        s2_ = coords2[:, 2].reshape(1, -1)

        adv = int(round(adv))

        # Advection
        s1 = s1 + adv * t1
        s2 = s2 + adv * t2

        nu = 1.0

        d2 = ((t1 - t2) ** 2) / range_time**2 + ((s1 - s2) ** 2) / range_space**2 + ((s1_ - s2_) ** 2) / range_space**2
        d = np.sqrt(d2 + 1e-12)

        K = (
            std**2
            * (2 ** (1 - nu))
            / gamma(nu)
            * (np.sqrt(2 * nu) * d) ** nu
            * kv(nu, np.sqrt(2 * nu) * d)
        )

        return K


class AdvectionModel(SimulationModel):
    def __init__(self):
        self._names = ['adv', 'range_time', 'range_space', 'std']
        self._names_alt = ["adv", r"$\rho_t$", r"$\rho_s$", r"$\sigma$"]
        self.lower_bound = (-3, 0.05, 0.05, 0.1)
        self.upper_bound = (4, 0.25, 0.25, 1.)

    @property
    def param_names(self):
        return self._names

    def get_param_names(self, alt_format=False):
        if alt_format:
            return self._names_alt
        return self.param_names

    @property
    def nb_params(self):
        return len(self.param_names)

    def to_title(self, vals, alt_format=False):
        if not alt_format:
            return fr"adv: {int(vals[0].round())} | $\rho_t$: {vals[1]:.2f}" + "\n" + fr"$\rho_s$: {vals[2]:.2f} | $\sigma$: {vals[3]:.2f}"
        else :
            return fr"adv: {int(vals[0].round())} | $\rho_t$: {vals[1]:.2f} | $\rho_s$: {vals[2]:.2f} | $\sigma$: {vals[3]:.2f}"

    def sample_thetas(self, n):
        return {
            'adv': np.random.randint(self.lower_bound[0], self.upper_bound[0], (n,)),
            'range_time': np.random.uniform(self.lower_bound[1], self.upper_bound[1], (n,)),
            'range_space': np.random.uniform(self.lower_bound[2], self.upper_bound[2], (n,)),
            'std': np.random.uniform(self.lower_bound[3], self.upper_bound[3], (n,))
        }
        # sampler = qmc.LatinHypercube(d=4)
        #
        # sample = sampler.random(n)
        #
        # l_bounds = self.lower_bound[:4]
        # u_bounds = self.upper_bound[:4]
        # sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        #
        # return {
        #     "adv": sample_scaled[:, 0].astype(int),
        #     "range_time": sample_scaled[:, 1],
        #     "range_space": sample_scaled[:, 2],
        #     "std": sample_scaled[:, 3],
        # }

    def simulate(self, p, t_size, s_dim, s_size, d_size):
        return _simulate_matern_grid_func(
            t_size, s_dim, s_size,
            p['adv'], p['range_time'], p['range_space'], p['std'],
            deepset_size=d_size
        )

    def get_cov(self, coords1, coords2, space_dim, theta_values):
        return matern_adv_cov(
            coords1, coords2, space_dim,
            theta_values[0], theta_values[1], theta_values[2], theta_values[3]
        )
        # return matern_adv_cov(coords1, coords2, space_dim, *theta_values)
