"""Shared pytest fixtures and configuration."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot is imported

import numpy as np
import pytest

from pypole import maps
from pypole.dipole import calculate_map


@pytest.fixture(scope="session")
def default_grid():
    """A 20x20 grid at 1 µm pixel size, shared for the whole test session."""
    return maps.get_grid((20, 20), 1e-6)


@pytest.fixture(scope="session")
def centred_dipole_map(default_grid):
    """A single vertical dipole at the centre of the default grid."""
    x, y = default_grid
    loc = np.array([[0.0, 0.0, 5e-6]])
    mom = np.array([[0.0, 0.0, 1e-14]])
    return calculate_map(x, y, loc, mom)
