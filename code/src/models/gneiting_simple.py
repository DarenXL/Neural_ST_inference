import numpy as np

import gstlearn as gl
from .base import SimulationModel
from scipy.special import gamma, kv
from scipy.stats import qmc

def _simulate_gneiting_matern_grid_func(time_size, space_dim, space_size, range_time, range_space, deepset_size=5):
    ns = 2000

    model = get_gst_model(space_dim, range_time, range_space)

    if space_dim == 1:

        dbgrid = gl.DbGrid.create(
            nx=[space_size, time_size],
            dx=[1. / space_size, 1. / time_size]
        )

        gl.simuSpectral(None, dbgrid, model, None, deepset_size, np.random.randint(10000), ns)

        # print("True coords", dbgrid['x1'],dbgrid["x2"])
        # print("True vals",dbgrid["z1"])

        batch = np.stack([dbgrid[i].reshape((space_size, time_size), order='F') for i in range(3, 3+deepset_size)]).transpose(0, 2, 1)

    elif space_dim == 2 :

        dbgrid = gl.DbGrid.create(
            nx=[space_size, space_size, time_size], dx=[1.0 / space_size, 1.0 / space_size, 1.0 / time_size]
        )

        gl.simuSpectral(
            None, dbgrid, model, None, deepset_size, np.random.randint(10000), ns
        )

        # print("True coords", dbgrid["x1"], dbgrid["x2"], dbgrid["x3"])
        # print("True vals", dbgrid["z1"])

        batch = np.stack(
            [
                dbgrid[i].reshape((space_size, space_size, time_size), order="F")
                for i in range(4, 4+deepset_size)
            ]
        ).transpose(0, 3, 2, 1)

    return np.expand_dims(np.stack(batch), 1)

def get_gst_model(space_dim, range_time, range_space):
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
        separability=1.,
    )

    # ctxtFac = gl.CovContext(1, gl.SpaceRN.create(space_dim + 1))
    # cov = gl.CorFactorized.create(ctxt=ctxtFac)
    # err = cov.addFactor(cov=fac, proj=proj_mat)

    model = gl.ModelGeneric()
    model.setCov(fac)

    return model


def matern_cov(
    coords_in, coords_out, y, space_dim, range_time, range_space
):

    model = get_gst_model(space_dim, range_time, range_space)

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


class GneitingModel(SimulationModel):
    def __init__(self, space_dim=1):
        self.space_dim = space_dim

        self._names = ["range_time", "range_space"]
        self._names_alt = [r"$\rho_t$", r"$\rho_s$"]
        self.lower_bound = (0.05, 0.05)
        self.upper_bound = (0.15, 0.5)

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

        return rf"$\rho_t$: {vals[0]:.2f} | $\rho_s$: {vals[1]:.2f}"

    def sample_thetas(self, n):
        return {
            "range_time": np.random.uniform(
                self.lower_bound[0], self.upper_bound[0], (n,)
            ),
            "range_space": np.random.uniform(
                self.lower_bound[1], self.upper_bound[1], (n,)
            ),
        }

    def simulate(self, p, t_size, s_dim, s_size, d_size):
        assert s_dim == self.space_dim
        return _simulate_gneiting_matern_grid_func(
            t_size,
            s_dim,
            s_size,
            p["range_time"],
            p["range_space"],
            deepset_size=d_size,
        )

    def kriging(self, coords1, coords2, y, space_dim, theta_values):
        return matern_cov(coords1, coords2, y, space_dim, *theta_values)

    def get_likelihood(self, coords, y, theta):

        model = get_gst_model(self.space_dim, *theta)

        dbgrid = gl.Db.create()
        names = []
        for i in range(len(coords[0])):
            dbgrid.addColumns(coords[:, i], radix="x" + str(i + 1))
            names.append("x" + str(i + 1))
        dbgrid.setLocators(names, gl.ELoc.X)
        # print("reconstructed coords",dbgrid["x1"], dbgrid["x2"], dbgrid["x3"])

        # print(y)

        dbgrid.addColumns(y, radix="z1")
        dbgrid.setLocator("z1", gl.ELoc.Z)
        # print("reconstructed vals",dbgrid['z1'])

        # print(dbgrid)

        return model.computeLogLikelihood(dbgrid)
