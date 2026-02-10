import numpy as np
import gstlearn as gl
from .base import SimulationModel

def _simulate_matern_grid_func(time_size, space_size, adv, range_time, range_space, std, deepset_size=5):
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

    return np.expand_dims(np.stack(batch), 1)

class AdvectionModel(SimulationModel):
    def __init__(self):
        self._names = ['adv', 'range_time', 'range_space', 'std']
        self.lower_bound = (-3, 0.05, 0.05, 0.1)
        self.upper_bound = (4, 0.25, 0.25, 1.)

    @property
    def param_names(self):
        return self._names

    def sample_thetas(self, n):
        return {
            'adv': np.random.randint(self.lower_bound[0], self.upper_bound[0], (n,)),
            'range_time': np.random.uniform(self.lower_bound[1], self.upper_bound[1], (n,)),
            'range_space': np.random.uniform(self.lower_bound[2], self.upper_bound[2], (n,)),
            'std': np.random.uniform(self.lower_bound[3], self.upper_bound[3], (n,))
        }

    def simulate(self, p, t_size, s_size, d_size):
        return _simulate_matern_grid_func(
            t_size, s_size,
            p['adv'], p['range_time'], p['range_space'], p['std'],
            deepset_size=d_size
        )