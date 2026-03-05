"""Tests for pypole.plotting module.

The Agg backend is activated in conftest.py before any matplotlib import.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pypole import plotting


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test to avoid memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def small_map():
    """A small 10x10 float64 map with a simple gradient."""
    rng = np.random.default_rng(0)
    return rng.uniform(-1e-6, 1e-6, (10, 10))


@pytest.fixture
def map_pair(small_map):
    """Two maps of the same shape for compare_maps tests."""
    rng = np.random.default_rng(1)
    fitted = small_map + rng.normal(0, 1e-8, small_map.shape)
    return small_map, fitted


# ---------------------------------------------------------------------------
# plot_map
# ---------------------------------------------------------------------------


class TestPlotMap:
    def test_returns_axes(self, small_map):
        import matplotlib.axes

        ax = plotting.plot_map(small_map)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_title_is_set(self, small_map):
        ax = plotting.plot_map(small_map, title="test title")
        assert ax.get_title() == "test title"

    def test_default_empty_title(self, small_map):
        ax = plotting.plot_map(small_map)
        assert ax.get_title() == ""

    def test_accepts_existing_axes(self, small_map):
        _, pre_ax = plt.subplots()
        returned_ax = plotting.plot_map(small_map, ax=pre_ax)
        assert returned_ax is pre_ax

    @pytest.mark.parametrize("shape", [(5, 5), (10, 20), (50, 50)])
    def test_various_map_shapes(self, shape):
        rng = np.random.default_rng(99)
        data = rng.uniform(-1e-6, 1e-6, shape)
        ax = plotting.plot_map(data)
        assert ax is not None

    def test_cbar_label_default(self, small_map):
        """Default colorbar label should be 'B [T]'."""
        ax = plotting.plot_map(small_map)
        # colorbar is attached to the figure — verify the axes has an image
        assert len(ax.get_images()) == 1

    @pytest.mark.parametrize("cbar_label", ["B [T]", "Bz [nT]", "field"])
    def test_custom_cbar_label(self, small_map, cbar_label):
        ax = plotting.plot_map(small_map, cbar_label=cbar_label)
        assert ax is not None


# ---------------------------------------------------------------------------
# compare_maps
# ---------------------------------------------------------------------------


class TestCompareMaps:
    def test_runs_without_error(self, map_pair, capsys):
        map1, map2 = map_pair
        plotting.compare_maps(map1, map2)

    def test_prints_residual_stats(self, map_pair, capsys):
        map1, map2 = map_pair
        plotting.compare_maps(map1, map2)
        captured = capsys.readouterr()
        assert "residual" in captured.out
        assert "std" in captured.out

    def test_prints_dipolarity(self, map_pair, capsys):
        map1, map2 = map_pair
        plotting.compare_maps(map1, map2)
        captured = capsys.readouterr()
        assert "dipolarity" in captured.out

    def test_title_passed(self, map_pair, capsys):
        map1, map2 = map_pair
        # Should not raise — title flows into fig.suptitle
        plotting.compare_maps(map1, map2, title="my title")

    @pytest.mark.parametrize("shape", [(5, 5), (20, 30)])
    def test_various_shapes(self, shape):
        rng = np.random.default_rng(7)
        m1 = rng.uniform(-1e-6, 1e-6, shape)
        m2 = rng.uniform(-1e-6, 1e-6, shape)
        plotting.compare_maps(m1, m2)  # must not raise
