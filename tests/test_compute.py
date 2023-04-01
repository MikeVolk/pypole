# type: ignore
import numpy as np
import pytest
from scipy.io import loadmat

from pypole import compute


def test_upward_continue():

    matlab = loadmat("tests/QDMlab_comparison_uc.mat")

    # Load data from matlab
    # map for upward continuation
    compare_bin = matlab["Bin"]
    # upward continued map from QDMlab.UpCont (10e-6 m)
    matlab_bout = matlab["Bout"]

    pixel_size = matlab["pixel_size"]
    distance = matlab["dh"]

    pypole_bout = compute.upward_continue(
        compare_bin, distance=distance, pixel_size=pixel_size
    )

    assert np.allclose(pypole_bout, matlab_bout)


def test_rms():
    np.random.seed(0)
    b_map_2d = np.random.randint(0,10, size=(10,10))
    expected_rms_2d = np.sqrt(np.mean(np.square(b_map_2d)))
    result_rms_2d = compute.rms(b_map_2d)
    np.testing.assert_almost_equal(result_rms_2d, expected_rms_2d)

    b_map_3d = np.random.randint(0,10, size=(5,10,10))
    expected_rms_3d = np.sqrt(np.mean(np.square(b_map_3d), axis=(1,2)))
    result_rms_3d = compute.rms(b_map_3d)
    np.testing.assert_almost_equal(result_rms_3d, expected_rms_3d)

    b_map_nd = np.random.randint(0,10, size=(2,2,10,10,10))
    expected_rms_nd = np.sqrt(np.mean(np.square(b_map_nd), axis=(-2,-1)))
    result_rms_nd = compute.rms(b_map_nd)
    np.testing.assert_almost_equal(result_rms_nd, expected_rms_nd)
