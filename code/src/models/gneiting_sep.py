import numpy as np

import gstlearn as gl
from .base import SimulationModel
from scipy.special import gamma, kv
from scipy.stats import qmc

def _simulate_gneiting_matern_grid_func(time_size, space_dim, space_size, sep, range_time, range_space, std, deepset_size=5):
    sep = np.maximum(sep, np.array(0.05))
    sep = np.minimum(sep, np.array(.95))
    range_time = np.maximum(range_time, np.array(0.05))
    range_space = np.maximum(range_space, np.array(0.05))
    std = np.maximum(std, np.array(0.1))

    typem = ["GNEITING_G", "GNEITING_M", "GNEITING_C"][1]
    ns = 2000

    alpha = 2.0
    beta = 1.0
    nu = 1.0
    rr = 1
    range_space = [range_space]*space_dim
    angles = [0]*space_dim

    ctxt = gl.CovContext(1, gl.SpaceRN.create(space_dim))

    fac = gl.CorGneiting.create(
        ctxt=ctxt,
        type=gl.ECov.fromKey(typem),
        alpha=alpha,
        beta=beta,
        timeRange=range_time,
        params=nu,
        kappas=rr,
        ranges=range_space,
        angles=angles,
        flagRange=False,
        separability=sep,
    )

    model = gl.ModelGeneric()
    model.setCov(fac)

    if space_dim == 1:

        dbgrid = gl.DbGrid.create(
            nx=[space_size, time_size],
            dx=[1. / space_size, 1. / time_size]
        )

        gl.simuSpectral(None, dbgrid, model, None, deepset_size, np.random.randint(10000), ns)

        batch = np.stack([dbgrid[i].reshape((space_size, time_size), order='F') for i in range(3, 3+deepset_size)]).transpose(0, 2, 1)

    elif space_dim == 2 :

        dbgrid = gl.DbGrid.create(
            nx=[space_size, space_size, time_size], dx=[1.0 / space_size, 1.0 / space_size, 1.0 / time_size]
        )

        gl.simuSpectral(
            None, dbgrid, model, None, deepset_size, np.random.randint(10000), ns
        )

        batch = np.stack(
            [
                dbgrid[i].reshape((space_size, space_size, time_size), order="F")
                for i in range(4, 4+deepset_size)
            ]
        ).transpose(0, 3, 2, 1)

    return np.expand_dims(np.stack(batch)*std, 1)


def matern_sep_cov(coords_in, coords_out, y, space_dim, sep, range_time, range_space, std):

    sep = np.maximum(sep, np.array(0.05))
    sep = np.minimum(sep, np.array(.95))
    range_time = np.maximum(range_time, np.array(0.05))
    range_space = np.maximum(range_space, np.array(0.05))
    std = np.maximum(std, np.array(0.1))

    typem = ["GNEITING_G", "GNEITING_M", "GNEITING_C"][1]
    ns = 2000

    alpha = 2.0
    beta = 1.0
    nu = 1.0
    rr = 1
    range_space = [range_space]*space_dim
    angles = [0]*space_dim

    ctxt = gl.CovContext(1, gl.SpaceRN.create(space_dim))

    fac = gl.CorGneiting.create(
        ctxt=ctxt,
        type=gl.ECov.fromKey(typem),
        alpha=alpha,
        beta=beta,
        timeRange=range_time,
        params=nu,
        kappas=rr,
        ranges=range_space,
        angles=angles,
        flagRange=False,
        separability=sep,
    )

    model = gl.ModelGeneric()
    model.setCov(fac)

    dbin = gl.Db.create()
    dbout = gl.Db.create()
    names = []
    for i in range(len(coords_in[0])):
        dbin.addColumns(coords_in[:, i], radix="x" + str(i + 1))
        dbout.addColumns(coords_out[:, i], radix="x" + str(i + 1))
        names.append("x" + str(i + 1))
    dbin.setLocators(names, gl.ELoc.X)
    dbout.setLocators(names, gl.ELoc.X)

    dbin.addColumns(y, radix="z1")
    dbin.setLocator("z1", gl.ELoc.Z)

    uniqueNeigh = gl.NeighUnique.create(space=gl.SpaceRN.create(space_dim + 1))

    err = gl.kriging(dbin, dbout, model, uniqueNeigh)
    return dbout['z1']


class GneitingSepModel(SimulationModel):
    def __init__(self, space_dim=1):
        self.space_dim = space_dim

        self._names = ["sep", "range_time", "range_space", "std"]
        self._names_alt = ["sep", r"$\rho_t$", r"$\rho_s$", r"$\sigma$"]
        self.lower_bound = (0.05, 0.05, 0.05, 0.1)
        self.upper_bound = (1.0, 0.15, 0.25, 1.0)

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
            return (
                rf"sep: {vals[0]:.2f} | $\rho_t$: {vals[1]:.2f}"
                + "\n"
                + rf"$\rho_s: {vals[2]:.2f} | $\sigma$: {vals[3]:.2f}"
            )
        else:
            return rf"sep: {vals[0]:.2f} | $\rho_t$: {vals[1]:.2f} | $\rho_s$: {vals[2]:.2f}| $\sigma$: {vals[3]:.2f}"

    def sample_thetas(self, n):
        return {
            "sep": np.random.uniform(self.lower_bound[0], self.upper_bound[0], (n,)),
            "range_time": np.random.uniform(
                self.lower_bound[1], self.upper_bound[1], (n,)
            ),
            "range_space": np.random.uniform(
                self.lower_bound[2], self.upper_bound[2], (n,)
            ),
            "std": np.random.uniform(self.lower_bound[3], self.upper_bound[3], (n,)),
        }

    def simulate(self, p, t_size, s_dim, s_size, d_size):
        assert s_dim == self.space_dim
        return _simulate_gneiting_matern_grid_func(
            t_size,
            s_dim,
            s_size,
            p["sep"],
            p["range_time"],
            p["range_space"],
            p["std"],
            deepset_size=d_size,
        )

    def kriging(self, coords1, coords2, y, space_dim, theta_values):
        return matern_sep_cov(coords1, coords2, y, space_dim, *theta_values)

    def get_likelihood(self, coords, y, theta):

        pass
