"""Tests for pypole.dipole module."""

import numpy as np
import pytest

from pypole import maps
from pypole.dipole import calculate_map, dipole_field, synthetic_map


# ---------------------------------------------------------------------------
# dipole_field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pixels, expected_shape",
    [
        ((10, 10), (10, 10)),
        ((5, 15), (15, 5)),
        ((1, 1), (1, 1)),
        ((30, 20), (20, 30)),
    ],
)
def test_dipole_field_output_shape(pixels, expected_shape):
    x, y = maps.get_grid(pixels, 1e-6)
    result = dipole_field(x, y, 0.0, 0.0, 5e-6, 0.0, 0.0, 1e-14)
    assert result.shape == expected_shape


def test_dipole_field_zero_moment_gives_zero(default_grid):
    x, y = default_grid
    result = dipole_field(x, y, 0.0, 0.0, 5e-6, 0.0, 0.0, 0.0)
    np.testing.assert_array_equal(result, 0.0)


def test_dipole_field_returns_float64(default_grid):
    x, y = default_grid
    result = dipole_field(x, y, 0.0, 0.0, 5e-6, 0.0, 0.0, 1e-14)
    assert result.dtype == np.float64


def test_dipole_field_centre_source_radial_symmetry(default_grid):
    """Bz of a centred vertical dipole is radially symmetric: B(r) == B(-r)."""
    x, y = default_grid
    result = dipole_field(x, y, 0.0, 0.0, 5e-6, 0.0, 0.0, 1e-14)
    np.testing.assert_allclose(result, result[::-1, ::-1], rtol=1e-10)


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
def test_dipole_field_scales_linearly_with_moment(default_grid, scale):
    """Field is proportional to dipole moment magnitude."""
    x, y = default_grid
    b_ref = dipole_field(x, y, 0.0, 0.0, 5e-6, 0.0, 0.0, 1e-14)
    b_scaled = dipole_field(x, y, 0.0, 0.0, 5e-6, 0.0, 0.0, scale * 1e-14)
    np.testing.assert_allclose(b_scaled, scale * b_ref, rtol=1e-10)


def test_dipole_field_decays_with_distance(default_grid):
    """Field magnitude must decrease as sensor moves further from the source."""
    x, y = default_grid
    b_close = dipole_field(x, y, 0.0, 0.0, 2e-6, 0.0, 0.0, 1e-14)
    b_far = dipole_field(x, y, 0.0, 0.0, 10e-6, 0.0, 0.0, 1e-14)
    assert np.abs(b_far).max() < np.abs(b_close).max()


# ---------------------------------------------------------------------------
# calculate_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_sources", [1, 3, 10])
def test_calculate_map_shape(default_grid, n_sources):
    x, y = default_grid
    locations = np.zeros((n_sources, 3))
    locations[:, 2] = 1e-6
    moments = np.zeros((n_sources, 3))
    moments[:, 2] = 1e-14
    result = calculate_map(x, y, locations, moments)
    assert result.shape == (20, 20)


def test_calculate_map_superposition(default_grid):
    """B from two sources must equal the sum of individual fields."""
    x, y = default_grid
    loc1 = np.array([[1e-6, 0.0, 2e-6]])
    loc2 = np.array([[-1e-6, 0.0, 2e-6]])
    m1 = np.array([[0.0, 0.0, 1e-14]])
    m2 = np.array([[0.0, 0.0, 2e-14]])

    b_combined = calculate_map(x, y, np.vstack([loc1, loc2]), np.vstack([m1, m2]))
    b_sum = calculate_map(x, y, loc1, m1) + calculate_map(x, y, loc2, m2)

    np.testing.assert_allclose(b_combined, b_sum, rtol=1e-10)


def test_calculate_map_sensor_distance_attenuates(default_grid):
    """Increasing sensor_distance reduces peak field amplitude."""
    x, y = default_grid
    locations = np.array([[0.0, 0.0, 0.0]])
    moments = np.array([[0.0, 0.0, 1e-14]])
    b_close = calculate_map(x, y, locations, moments, sensor_distance=1e-6)
    b_far = calculate_map(x, y, locations, moments, sensor_distance=10e-6)
    assert np.abs(b_far).max() < np.abs(b_close).max()


def test_calculate_map_matches_dipole_field(default_grid):
    """Single-source calculate_map must match direct dipole_field call."""
    x, y = default_grid
    x_src, y_src, z_src = 0.0, 0.0, 5e-6
    mx, my, mz = 0.0, 0.0, 1e-14
    sensor_distance = 0.0

    locations = np.array([[x_src, y_src, z_src]])
    moments = np.array([[mx, my, mz]])
    b_map = calculate_map(x, y, locations, moments, sensor_distance=sensor_distance)
    b_direct = dipole_field(x, y, x_src, y_src, z_src, mx, my, mz)

    np.testing.assert_allclose(b_map, b_direct, rtol=1e-10)


# ---------------------------------------------------------------------------
# synthetic_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_sources, pixels",
    [
        (1, (10, 10)),
        (5, (20, 30)),
        (10, (15, 15)),
    ],
)
def test_synthetic_map_shape(n_sources, pixels):
    np.random.seed(0)
    result = synthetic_map(n_sources=n_sources, pixels=pixels)
    assert result.shape == (pixels[1], pixels[0])


def test_synthetic_map_default_shape():
    np.random.seed(0)
    result = synthetic_map()
    assert result.shape == (100, 100)


def test_synthetic_map_returns_float64():
    np.random.seed(0)
    result = synthetic_map(n_sources=2, pixels=(5, 5))
    assert result.dtype == np.float64


def test_synthetic_map_reproducible():
    """Same seed must yield the same map."""
    np.random.seed(42)
    map1 = synthetic_map(n_sources=3, pixels=(10, 10))
    np.random.seed(42)
    map2 = synthetic_map(n_sources=3, pixels=(10, 10))
    np.testing.assert_array_equal(map1, map2)


def test_synthetic_map_nonzero():
    np.random.seed(0)
    result = synthetic_map(n_sources=1, pixels=(10, 10))
    assert np.any(result != 0.0)
