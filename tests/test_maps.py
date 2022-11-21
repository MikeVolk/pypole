import pytest
from pypole import maps
import numpy as np

@pytest.mark.parametrize("test_input,expected",
                         [({(10,10),1}, np.array())])

def test_eval(test_input, expected):
    assert maps(test_input) == expected
