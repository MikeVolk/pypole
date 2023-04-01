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



def test_dipolarity_param():
    np.random.seed(0)

    # Test for maps with shape (10, 10)
    data_map_2d = np.random.randint(0, 100, size=(10, 10))
    fitted_map_2d = np.random.randint(0, 100, size=(10, 10))
    dp_2d = compute.dipolarity_param(data_map_2d, fitted_map_2d)

    # Test for maps with shape (2, 10, 10)
    data_map_3d = np.random.randint(0, 100, size=(2, 10, 10))
    fitted_map_3d = np.random.randint(0, 100, size=(2, 10, 10))
    dp_3d = compute.dipolarity_param(data_map_3d, fitted_map_3d)

    # Test for maps with shape (2, 3, 4, 10, 10)
    data_map_5d = np.random.randint(0, 100, size=(2, 3, 4, 10, 10))
    fitted_map_5d = np.random.randint(0, 100, size=(2, 3, 4, 10, 10))
    dp_5d = compute.dipolarity_param(data_map_5d, fitted_map_5d)

    # Check that dipolarity parameters are within the valid range [0, 1]
    assert 0 <= dp_2d <= 1
    assert np.all((0 <= dp_3d) & (dp_3d <= 1))
    assert np.all((0 <= dp_5d) & (dp_5d <= 1))

    # Test for mismatched shapes
    with pytest.raises(ValueError):
        compute.dipolarity_param(data_map_2d, fitted_map_3d)


def test_pad_map():

    # Test 2D maps
    b_map = np.random.randint(0, 10, size=(3, 3))
    padded = compute.pad_map(b_map, oversample=2)
    assert padded.shape == (9,9)
    assert np.array_equal(padded[3:6, 3:6], b_map)

    b_map = np.random.randint(0, 10, size=(4, 3))
    padded = compute.pad_map(b_map, oversample=3)
    assert padded.shape == (20, 15)
    assert np.array_equal(padded[8:8+4, 6:6+3], b_map)

    # Test ND maps
    b_map = np.random.randint(0, 10, size=(2, 3, 3))
    padded = compute.pad_map(b_map, oversample=2)
    assert padded.shape == (2, 9, 9)
    assert np.array_equal(padded[:, 3:6, 3:6], b_map)

    b_map = np.random.randint(0, 10, size=(2, 3, 4))
    padded = compute.pad_map(b_map, oversample=3)
    assert padded.shape == (2, 15, 20)
    assert np.array_equal(padded[:, 6:6+3, 8:8+4], b_map)

    b_map = np.random.randint(0, 10, size=(2, 3, 4, 5, 6))
    padded = compute.pad_map(b_map, oversample=4)
    assert padded.shape == (2, 3, 4, 35, 42)
    assert np.array_equal(padded[:, :, :, 15:15+5, 18:18+6], b_map)