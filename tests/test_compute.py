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
