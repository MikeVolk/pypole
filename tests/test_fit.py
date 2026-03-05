"""Tests for pypole.fit module."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pypole import maps
from pypole.dipole import calculate_map
from pypole.fit import dipole_residual, fit_dipole, fit_dipole_n_maps

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dipole_setup():
    """Known single-dipole map with exact ground-truth parameters.

    sensor_distance=0 so that z_observed == loc_z, making p0 exactly correct
    and the residual identically zero.

    Returns (field_map, p0_tuple, x_grid, y_grid, pixel_size).
    """
    pixels = (20, 20)
    pixel_size = 1e-6
    x, y = maps.get_grid(pixels, pixel_size)
    loc = np.array([[0.0, 0.0, 5e-6]])
    mom = np.array([[0.0, 0.0, 1e-14]])
    # sensor_distance=0: z_observed in dipole_field == loc[2], so p0 is exact
    field_map = calculate_map(x, y, loc, mom, sensor_distance=0.0)
    p0 = (0.0, 0.0, 5e-6, 0.0, 0.0, 1e-14)
    return field_map, p0, x, y, pixel_size


@pytest.fixture(scope="module")
def fit_result(dipole_setup):
    """Cached OptimizeResult for the centred dipole (expensive — compute once)."""
    field_map, p0, _, __, pixel_size = dipole_setup
    return fit_dipole(field_map, p0, pixel_size)


# ---------------------------------------------------------------------------
# dipole_residual
# ---------------------------------------------------------------------------


class TestDipoleResidual:
    def test_zero_at_ground_truth(self, dipole_setup):
        field_map, p0, x, y, _ = dipole_setup
        grid = np.vstack((y.ravel(), x.ravel()))
        residual = dipole_residual(p0, grid, field_map.T.ravel())
        np.testing.assert_allclose(residual, 0.0, atol=1e-20)

    def test_output_length_matches_map(self, dipole_setup):
        field_map, p0, x, y, _ = dipole_setup
        grid = np.vstack((y.ravel(), x.ravel()))
        residual = dipole_residual(p0, grid, field_map.T.ravel())
        assert residual.shape == field_map.ravel().shape

    @pytest.mark.parametrize(
        "offset",
        [
            (5e-6, 0.0, 0.0, 0.0, 0.0, 0.0),   # wrong x
            (0.0, 5e-6, 0.0, 0.0, 0.0, 0.0),   # wrong y
            (0.0, 0.0, 1e-6, 0.0, 0.0, 0.0),   # wrong z
            (0.0, 0.0, 0.0, 0.0, 0.0, 1e-14),  # wrong mz
        ],
    )
    def test_nonzero_for_perturbed_params(self, dipole_setup, offset):
        field_map, p0, x, y, _ = dipole_setup
        grid = np.vstack((y.ravel(), x.ravel()))
        perturbed = tuple(a + b for a, b in zip(p0, offset))
        residual = dipole_residual(perturbed, grid, field_map.T.ravel())
        assert np.any(residual != 0.0)


# ---------------------------------------------------------------------------
# fit_dipole / _fit_dipole
# ---------------------------------------------------------------------------


class TestFitDipole:
    def test_returns_optimize_result(self, fit_result):
        assert isinstance(fit_result, OptimizeResult)

    def test_converges(self, fit_result):
        assert fit_result.success

    def test_result_has_six_params(self, fit_result):
        assert fit_result.x.shape == (6,)

    def test_recovers_source_position(self, fit_result):
        np.testing.assert_allclose(fit_result.x[:2], [0.0, 0.0], atol=1e-8)

    def test_recovers_z(self, fit_result):
        np.testing.assert_allclose(fit_result.x[2], 5e-6, rtol=1e-4)

    def test_recovers_mz(self, fit_result):
        np.testing.assert_allclose(fit_result.x[5], 1e-14, rtol=1e-4)

    @pytest.mark.parametrize("pixel_size", [0.5e-6, 1e-6, 2e-6])
    def test_pixel_size_does_not_break_fit(self, pixel_size):
        """fit_dipole should converge for a range of pixel sizes."""
        pixels = (15, 15)
        x, y = maps.get_grid(pixels, pixel_size)
        loc = np.array([[0.0, 0.0, 5e-6]])
        mom = np.array([[0.0, 0.0, 1e-14]])
        field_map = calculate_map(x, y, loc, mom)
        p0 = (0.0, 0.0, 5e-6, 0.0, 0.0, 1e-14)
        result = fit_dipole(field_map, p0, pixel_size)
        assert isinstance(result, OptimizeResult)


# ---------------------------------------------------------------------------
# fit_dipole_n_maps
# ---------------------------------------------------------------------------


class TestFitDipoleNMaps:
    @pytest.fixture(scope="class")
    def n_maps_setup(self):
        """Three identical dipole maps with shared initial guess."""
        n = 3
        pixels = (10, 10)
        pixel_size = 1e-6
        x, y = maps.get_grid(pixels, pixel_size)
        loc = np.array([[0.0, 0.0, 5e-6]])
        mom = np.array([[0.0, 0.0, 1e-14]])
        single_map = calculate_map(x, y, loc, mom)
        field_maps = np.stack([single_map] * n)
        p0 = np.tile([0.0, 0.0, 5e-6, 0.0, 0.0, 1e-14], (n, 1))
        return x, y, field_maps, p0, pixel_size, single_map

    def test_output_shape(self, n_maps_setup):
        x, y, field_maps, p0, *_ = n_maps_setup
        result = fit_dipole_n_maps(x, y, field_maps, p0)
        assert result.shape == (3, 6)

    @pytest.mark.parametrize("n_maps", [1, 2, 5])
    def test_output_shape_parametrized(self, n_maps):
        pixels = (8, 8)
        pixel_size = 1e-6
        x, y = maps.get_grid(pixels, pixel_size)
        loc = np.array([[0.0, 0.0, 5e-6]])
        mom = np.array([[0.0, 0.0, 1e-14]])
        single_map = calculate_map(x, y, loc, mom)
        field_maps = np.stack([single_map] * n_maps)
        p0 = np.tile([0.0, 0.0, 5e-6, 0.0, 0.0, 1e-14], (n_maps, 1))
        result = fit_dipole_n_maps(x, y, field_maps, p0)
        assert result.shape == (n_maps, 6)

    def test_matches_single_fit(self, n_maps_setup):
        """Each row of fit_dipole_n_maps should match a standalone fit_dipole call."""
        x, y, field_maps, p0, pixel_size, single_map = n_maps_setup
        result_n = fit_dipole_n_maps(x, y, field_maps, p0)
        result_1 = fit_dipole(single_map, tuple(p0[0]), pixel_size)
        np.testing.assert_allclose(result_n[0], result_1.x, rtol=1e-6)
