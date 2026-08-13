import numpy as np

import gstlearn as gl
from .base import SimulationModel
from scipy.special import gamma, kv
from scipy.stats import qmc

def _simulate_gneiting_matern_grid_func(time_size, space_dim, space_size, sep, range_time, range_space, seed=0):
    ns = 4000

    model = get_gst_model(space_dim, sep, range_time, range_space)

    if space_dim == 1:

        dbgrid = gl.DbGrid.create(
            nx=[space_size, time_size],
            dx=[1. / space_size, 1. / time_size]
        )

        gl.simuSpectral(None, dbgrid, model, None, 1, seed, ns)

        batch = dbgrid[3].reshape((space_size, time_size), order="F").transpose(1, 0)

    elif space_dim == 2 :

        dbgrid = gl.DbGrid.create(
            nx=[space_size, space_size, time_size], dx=[1.0 / space_size, 1.0 / space_size, 1.0 / time_size]
        )

        gl.simuSpectral(
            None, dbgrid, model, None, 1, seed, ns
        )

        batch = dbgrid[4].reshape((space_size, space_size, time_size), order="F").transpose(2, 1, 0)

    return np.expand_dims(batch, (0,1))

def get_gst_model(space_dim, sep, range_time, range_space):
    sep = np.maximum(sep, np.array(0.))
    sep = np.minimum(sep, np.array(1.))
    range_time = np.maximum(range_time, np.array(0.05))
    range_space = np.maximum(range_space, np.array(0.05))

    typem = ["GNEITING_G", "GNEITING_M", "GNEITING_C"][1]

    alpha = 2.0
    beta = 1.0
    nu = 1.5
    rr = 1
    range_space = [range_space] * space_dim
    angles = [0] * space_dim

    ctxt = gl.CovContext(1, gl.SpaceRN.create(space_dim))

    # proj_mat = np.eye(space_dim + 1)
    # proj_mat[0:space_dim, space_dim] = 0

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

    ctxtFac = gl.CovContext(1, gl.SpaceRN.create(space_dim + 1))
    cov = gl.CorFactorized.create(ctxt=ctxtFac)
    # err = cov.addFactor(cov=fac, proj=proj_mat)

    model = gl.ModelGeneric()
    model.setCov(fac)

    return model


def matern_sep_cov(
    coords_in, coords_out, y, space_dim, sep, range_time, range_space
):

    model = get_gst_model(space_dim, sep, range_time, range_space)

    dbin = gl.Db.create()
    dbout = gl.Db.create()
    names = []
    for i in range(len(coords_in[0])):
        dbin.addColumns(coords_in[:, 1-i], radix="x" + str(i + 1))
        dbout.addColumns(coords_out[:, 1-i], radix="x" + str(i + 1))
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

        self._names = ["sep", "range_time", "range_space"]
        self._names_alt = [r"$\gamma$", r"$\rho_t$", r"$\rho_s$"]
        self.lower_bound = (0., 0.05, 0.05)
        self.upper_bound = (1., 0.15, 0.5)

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
                rf"$\gamma$: {vals[0]:.2f} | $\rho_t$: {vals[1]:.2f}"
                + "\n"
                + rf"$\rho_s: {vals[2]:.2f}"
            )
        else:
            return rf"$\gamma$: {vals[0]:.2f} | $\rho_t$: {vals[1]:.2f} | $\rho_s$: {vals[2]:.2f}"

    def sample_thetas(self, n):
        return {
            "sep": np.random.uniform(self.lower_bound[0], self.upper_bound[0], (n,)),
            "range_time": np.random.uniform(
                self.lower_bound[1], self.upper_bound[1], (n,)
            ),
            "range_space": np.random.uniform(
                self.lower_bound[2], self.upper_bound[2], (n,)
            ),
        }

    def simulate(self, p, t_size, s_dim, s_size, seed=0):
        assert s_dim == self.space_dim

        return _simulate_gneiting_matern_grid_func(
            t_size,
            s_dim,
            s_size,
            p["sep"],
            p["range_time"],
            p["range_space"],
            seed=seed
        )

    def eval_cov(self, coords, y, space_dim, theta_values):
        model = get_gst_model(self.space_dim, *theta_values)

        dbgrid = gl.Db.create()
        names = []
        for i in range(len(coords[0])):
            dbgrid.addColumns(coords[:, i], radix="x" + str(i + 1))
            names.append("x" + str(i + 1))
        dbgrid.setLocators(names, gl.ELoc.X)

        dbgrid.addColumns(y, radix="z1")
        dbgrid.setLocator("z1", gl.ELoc.Z)

    def kriging(self, coords1, coords2, y, space_dim, theta_values):
        return matern_sep_cov(coords1, coords2, y, space_dim, *theta_values)

    def get_likelihood(self, coords, y, theta):

        model = get_gst_model(self.space_dim, *theta)

        dbgrid = gl.Db.create()
        names = []
        for i in range(len(coords[0])):
            dbgrid.addColumns(coords[:, i], radix="x" + str(i + 1))
            names.append("x" + str(i + 1))
        dbgrid.setLocators(names, gl.ELoc.X)

        # print(y)

        dbgrid.addColumns(y, radix="z1")
        dbgrid.setLocator("z1", gl.ELoc.Z)

        # print(dbgrid)

        return model.computeLogLikelihood(dbgrid)
