import numpy as np

import gstlearn as gl
from .base import SimulationModel
from scipy.special import gamma, kv
from scipy.stats import qmc

def _simulate_gneiting_matern_grid_func(time_size, space_dim, space_size, adv, range_time, range_space, std, deepset_size=5):
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
    sep = 1.

    # speed = adv.copy()
    # adv = adv/np.sqrt(np.sum(adv**2)) # direction of advection
    proj_mat = np.eye(space_dim+1)
    proj_mat[0:space_dim, space_dim] = -np.array(adv)

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

    ctxtFac = gl.CovContext(1, gl.SpaceRN.create(space_dim+1))
    cov  = gl.CorFactorized.create(ctxt = ctxtFac)
    err = cov.addFactor(cov = fac, proj = proj_mat)

    model = gl.ModelGeneric()
    model.setCov(cov)

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

def get_gst_model(space_dim, adv, range_time, range_space, std):
    range_time = np.maximum(range_time, np.array(0.05))
    range_space = np.maximum(range_space, np.array(0.05))
    std = np.maximum(std, np.array(0.1))

    typem = ["GNEITING_G", "GNEITING_M", "GNEITING_C"][1]
    ns = 2000

    alpha = 2.0
    beta = 1.0
    nu = 1.0
    rr = 1
    range_space = [range_space] * space_dim
    angles = [0] * space_dim
    sep = 1.0

    # speed = adv.copy()
    # adv = adv/np.sqrt(np.sum(adv**2)) # direction of advection
    proj_mat = np.eye(space_dim + 1)
    proj_mat[0:space_dim, space_dim] = -np.array(adv)

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

    ctxtFac = gl.CovContext(1, gl.SpaceRN.create(space_dim + 1))
    cov = gl.CorFactorized.create(ctxt=ctxtFac)
    err = cov.addFactor(cov=fac, proj=proj_mat)

    model = gl.ModelGeneric()
    model.setCov(cov)

    return model

def matern_adv_cov(coords_in, coords_out, y, space_dim, adv, range_time, range_space, std):

    model = get_gst_model(space_dim, adv, range_time, range_space, std)

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


class GneitingAdvModel(SimulationModel):
    def __init__(self, space_dim=1):
        self.space_dim = space_dim

        if self.space_dim == 1 :
            self._names = ['adv', 'range_time', 'range_space', 'std']
            self._names_alt = ["adv", r"$\rho_t$", r"$\rho_s$", r"$\sigma$"]
            self.lower_bound = (-1, 0.05, 0.05, 0.1)
            self.upper_bound = (1, 0.15, 0.25, 1.)

        elif self.space_dim == 2 :
            self._names = ['adv1', 'adv2', 'range_time', 'range_space', 'std']
            self._names_alt = ["adv1", "adv2", r"$\rho_t$", r"$\rho_s$", r"$\sigma$"]
            self.lower_bound = (-1, -1, .05, 0.05, 0.1)
            self.upper_bound = (1, 1, .25, 0.25, 1.)

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
        if self.space_dim == 1 :
            if not alt_format:
                return fr"adv: {vals[0]:.2f} | $\rho_t$: {vals[1]:.2f}" + "\n" + fr"$\rho_s: {vals[2]:.2f} | $\sigma$: {vals[3]:.2f}"
            else :
                return fr"adv: {vals[0]:.2f} | $\rho_t$: {vals[1]:.2f} | $\rho_s$: {vals[2]:.2f}| $\sigma$: {vals[3]:.2f}"
        elif self.space_dim == 2 :
            if not alt_format:
                return fr"adv1: {vals[0]:.2f} | adv2: {vals[1]:.2f} | $\rho_t$: {vals[2]:.2f}" + "\n" + fr"$\rho_s$: {vals[3]:.2f} | $\sigma$: {vals[4]:.2f}"
            else :
                return fr"adv1: {vals[0]:.2f} | adv2: {vals[1]:.2f} | $\rho_t$: {vals[2]:.2f} | $\rho_s$: {vals[3]:.2f} | $\sigma$: {vals[4]:.2f}"

    def sample_thetas(self, n):
        if self.space_dim == 1 :
            return {
                "adv": np.random.uniform(
                    self.lower_bound[0], self.upper_bound[0], (n,)
                ),
                "range_time": np.random.uniform(
                    self.lower_bound[1], self.upper_bound[1], (n,)
                ),
                "range_space": np.random.uniform(
                    self.lower_bound[2], self.upper_bound[2], (n,)
                ),
                "std": np.random.uniform(
                    self.lower_bound[3], self.upper_bound[3], (n,)
                ),
            }
        elif self.space_dim == 2 :
            return {
                'adv1': np.random.uniform(self.lower_bound[0], self.upper_bound[0], (n,)),
                'adv2': np.random.uniform(self.lower_bound[1], self.upper_bound[1], (n,)),
                'range_time': np.random.uniform(self.lower_bound[2], self.upper_bound[2], (n,)),
                'range_space': np.random.uniform(self.lower_bound[3], self.upper_bound[3], (n,)),
                'std': np.random.uniform(self.lower_bound[4], self.upper_bound[4], (n,))
            }

    def simulate(self, p, t_size, s_dim, s_size, d_size):
        assert s_dim == self.space_dim
        if self.space_dim == 1 :
            return _simulate_gneiting_matern_grid_func(
                t_size, s_dim, s_size,
                p['adv'], p['range_time'], p['range_space'], p['std'],
                deepset_size=d_size
            )
        elif self.space_dim == 2 :
            return _simulate_gneiting_matern_grid_func(
                t_size, s_dim, s_size,
                [p['adv1'], p['adv2']], p['range_time'], p['range_space'], p['std'],
                deepset_size=d_size
            )

    def kriging(self, coords_in, coords_out, y, space_dim, theta_values):
        if self.space_dim == 1 :
            return matern_adv_cov(
                coords_in, coords_out, y, space_dim,
                theta_values[0], theta_values[1], theta_values[2], theta_values[3]
            )
        elif self.space_dim == 2 :
            return matern_adv_cov(
                coords_in, coords_out, y, space_dim,
                [theta_values[0], theta_values[1]], theta_values[2], theta_values[3], theta_values[4]
            )

    def get_likelihood(self, coords, y, theta):
        model = get_gst_model(self.space_dim, *theta)

        dbgrid = gl.Db.create()
        names = []
        for i in range(len(coords[0])):
            dbgrid.addColumns(coords[:, i], radix="x" + str(i + 1))
            names.append("x" + str(i + 1))
        dbgrid.setLocators(names, gl.ELoc.X)

        dbgrid.addColumns(y, radix="z1")
        dbgrid.setLocator("z1", gl.ELoc.Z)

        ndat = 100
        v = np.arange(dbgrid.getNSample())
        np.random.shuffle(v)
        inddat = v[0:ndat]
        dat = gl.Db.createReduce(dbgrid, ranks=inddat)

        return model.computeLogLikelihood(dat)
