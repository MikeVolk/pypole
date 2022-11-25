# type: ignore
import numpy as np
import pytest

from pypole import convert

dim2xyz_data = [
    [(1, 0, 0), (90, 0, 1)],
    [(0, 1, 0), (180, 0, 1)],
    [
        (0, 0, 1),
        (90, -90, 1),
    ],  # for x=0 and y=0, declination is not defined, here. Returns 90
    [(0, 0, -1), (90, 90, 1)],  # declination is not defined, here. Returns 90
    [(1, 1, 0), (135, 0, np.sqrt(2))],
    [(1, 1, 1), (135, -35.26438968, np.sqrt(3))],
    [[1.2, 3.4, 5.6], convert.xyz2dim([1.2, 3.4, 5.6])],
]


@pytest.mark.parametrize("xyz, dim", dim2xyz_data)
def test_dim2xyz(xyz, dim):
    assert np.allclose(convert.dim2xyz(dim), xyz)


z2inc_test_data = [
    (0, 1, 0),
    (0, -1, 0),
    (1, 1, -90),
    (-1, 1, 90),
    (0.5, 1, -30),
    (1, np.sqrt(2), -45),
]  # z, mag, inc


@pytest.mark.parametrize("z, m, inc", z2inc_test_data)
def test_z2inc(z, m, inc):
    assert np.allclose(convert.z2inc(z=z, mag=m), inc)


xy2dec_test_data = [(1, 0, 90), (0, 1, 180), (-1, 0, 270), (0, -1, 0)]  # x, y, dec


@pytest.mark.parametrize("x, y, dec", xy2dec_test_data)
def test_xy2dec(x, y, dec):
    assert convert.xy2dec(np.array([x, y])) == dec


xyz2dim_data = [
    [(1, 0, 0), (90, 0, 1)],
    [(0, 1, 0), (180, 0, 1)],
    [(0, 0, 1), (90, -90, 1)],  # declination is not defined, here. Returns 90 anyways
    [(0, 0, -1), (90, 90, 1)],
    [(1, 1, 0), (135, 0, np.sqrt(2))],
    [(1, 1, 1), (135, -35.26438968, np.sqrt(3))],
    [convert.dim2xyz([11.2, 33.21, 1.828e-6]), (11.2, 33.21, 1.828e-6)],
]  # xyz, dim


@pytest.mark.parametrize("xyz, dim", xyz2dim_data)
def test_xyz2dim(xyz, dim):
    assert np.allclose(convert.xyz2dim(xyz), dim)
