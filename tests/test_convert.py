# type: ignore
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pypole import convert


def test_dim2xyz():
    dim = np.array([[0, 0, 1]])
    assert_allclose(convert.dim2xyz(dim), [[0, -1, 0]])

    dim = np.array([[0, 90, 1]])
    assert_allclose(convert.dim2xyz(dim), [[0, 0, -1]], atol=1e-10)

    dim = np.array([[0, 0, 1], [0, 90, 1]])
    assert_allclose(convert.dim2xyz(dim), [[0, -1, 0], [0, 0, -1]], atol=1e-10)
