# type: ignore
import numpy as np

from pypole import maps

PIXELSIZE = 0.5


def test_grid2x2():
    # equality between one pixel and two pixel values
    assert np.allclose(maps.get_grid(2, 1), maps.get_grid((2, 2), 1))

    # explicit values
    assert np.all(maps.get_grid((2, 2), 1)[0] == np.array([[-1, 1], [-1, 1]]))
    assert np.all(maps.get_grid((2, 2), 1)[1] == np.array([[-1, -1], [1, 1]]))

    # including pixelsize
    assert np.all(
        maps.get_grid((2, 2), PIXELSIZE)[1] == PIXELSIZE * np.array([[-1, -1], [1, 1]])
    )


def test_grid6x6():
    # equality between one pixel and two pixel values
    assert np.allclose(maps.get_grid(6, 1), maps.get_grid((6, 6), 1))

    # explicit values
    assert np.all(maps.get_grid((6, 6), 1)[0][0, :] == np.linspace(-3, 3, 6))
    assert np.all(maps.get_grid((6, 6), 1)[1][:, 1] == np.linspace(-3, 3, 6))

    assert np.all(
        maps.get_grid((6, 6), PIXELSIZE)[1][:, 1] == PIXELSIZE * np.linspace(-3, 3, 6)
    )


def test_grid_rectangular():
    # explicit values
    assert np.all(maps.get_grid((2, 6), 1)[0][:, 0] == np.ones(6) * -1)
    assert np.all(maps.get_grid((2, 6), 1)[0][:, 1] == np.ones(6))

    assert np.all(maps.get_grid((2, 6), 1)[1][:, 0] == np.linspace(-3, 3, 6))
    assert np.all(maps.get_grid((2, 6), 1)[1][:, 1] == np.linspace(-3, 3, 6))

    assert np.all(
        maps.get_grid((2, 6), PIXELSIZE)[1][:, 1] == PIXELSIZE * np.linspace(-3, 3, 6)
    )


def test_get_random_locations():
    locs = maps.get_random_locations(100, (-1, 1), (-1, 1), (-1, 1))
    assert locs.shape == (100, 3)


def test_get_random_dim():
    dim = maps.get_random_dim(100, (-1, 1))
    assert dim.shape == (100, 3)
    assert np.all(dim[:, 0] < 360)
    assert np.all(dim[:, 1] <= 90)


def test_get_random_sources():
    locs, sources = maps.get_random_sources(100, (-1, 1), (-1, 1), (-1, 1), (-1, 1))
    assert sources.shape == (100, 3)
    assert locs.shape == (100, 3)
    assert np.all(sources[:, 0] < 360)
    assert np.all(sources[:, 1] <= 90)
