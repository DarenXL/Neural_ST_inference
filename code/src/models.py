import gstlearn as gl
import gstlearn.plot as gp
import gstlearn.document as gdoc
gdoc.setNoScroll()


def simulate_matern_grid(time_size, space_size, adv, range_time, range_space, std, deepset_size=5):
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