import typing

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from pypole import NDArray64, compute, maps
from pypole.dipole import dipole_field


def fit_dipole_n_maps(
    x_grid: NDArray64,
    y_grid: NDArray64,
    b_maps: NDArray64,
    initial_guess: NDArray64,
) -> NDArray64:
    """fits a series of maps each with a single dipole.

    Parameters
    ----------
    x_grid : ndarray(pixel, pixel)
        x grid
    y_grid : ndarray(pixel, pixel)
        y grid
    b_maps : ndarray(n_maps, pixel, pixel)
        magnetic field map for all frames
    initial_guess : ndarray(n_maps, 3)
        initial guess for dipole parameters
    """
    n_maps = b_maps.shape[0]
    best_fit_dipoles = np.empty((n_maps, 6))

    for map_index in numba.prange(n_maps):
        best_fit_dipoles[map_index, :] = _fit_dipole(
            b_map=b_maps[map_index],
            p0=initial_guess[map_index],
            x_grid=x_grid,
            y_grid=y_grid,
        )
    return best_fit_dipoles


def __initial_guess_from_synthetic(mvec: NDArray64) -> NDArray64:
    """
    Get initial guess for dipolarity parameter calculation

    Parameters
    ----------
    mvec: ndarray(3,)
        magnetic vector

    Returns
    -------
    initial guess: ndarray(6,)
        initial guess for dipolarity parameter calculation
    """
    return np.array([0, 0, 5e-6, mvec[0], mvec[1], mvec[2]])


def dipole_residual(
    params: tuple[float, float, float, float, float, float],
    grid: NDArray64,
    data: NDArray64,
) -> NDArray64:
    """residual function for fitting a single dipole to a magnetic field map

    Parameters
    ----------
    params: tuple(floats)
        parameters for field calculation:
            [x_source, y_source, z_source, mx, my, mz]
    grid: ndarray
        x,y grid
    data: ndarray
        field data

    Returns
    -------
        difference of calculated map from params and field data
    """
    x, y = grid
    arr = dipole_field(
        x_grid=x,
        y_grid=y,
        x_source=params[0],
        y_source=params[1],
        z_observed=params[2],
        mx=params[3],
        my=params[4],
        mz=params[5],
    )
    return arr.ravel() - data


def fit_dipole(b_map, p0, pixel_size=1):
    """fits a single dipole to a magnetic field map

    Parameters
    ----------
    b_map: ndarray
        magnetic field map in Tesla
    p0: tuple(floats)
        initial guess for dipole parameters
    pixel_size: float
        pixel size in meters

    Returns
    -------
    dipole parameters: tuple(floats)
        dipole parameters [x_source, y_source, z_source, mx, my, mz]
    """

    x_grid, y_grid = maps.get_grid(pixels=b_map.shape, pixel_size=pixel_size)
    return _fit_dipole(b_map, p0, x_grid, y_grid)


def _fit_dipole(b_map, p0, x_grid, y_grid):
    grid = np.vstack((y_grid.ravel(), x_grid.ravel()))
    return least_squares(
        dipole_residual,
        p0,
        args=(grid, b_map.T.ravel()),
        loss="huber",
        method="trf",
        gtol=2.3e-16,
        ftol=2.3e-16,
        xtol=2.3e-16,
        max_nfev=5000,
    )
