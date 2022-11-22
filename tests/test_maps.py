# type: ignore
import numpy as np
import pytest

from pypole import maps


@pytest.mark.parametrize("test_input,expected", [({(10, 10), 1}, np.array())])
def test_eval(test_input, expected):
    assert maps(test_input) == expected
