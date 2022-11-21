import typing

import numba
import numpy as np
import tqdm
from scipy.optimize import least_squares

from pypole import compute
from pypole.dipole import dipole_field


def fit_dipole_n_maps(x_grid, y_grid, b_maps, initial_guess):
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

    with tqdm.tqdm(total=n_maps, unit="fit", ncols=80) as pbar:
        for map_index in numba.prange(n_maps):
            best_fit_dipoles[map_index, :] = fit_dipole(
                b_map=b_maps[map_index],
                p0=initial_guess[map_index],
                x_grid=x_grid,
                y_grid=y_grid,
            )
            pbar.update(1)
    return best_fit_dipoles


@numba.jit(nopython=True)
def __initial_guess_from_synthetic(mvec):
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
    return [0, 0, 5e-6, mvec[0], mvec[1], mvec[2]]


def dipole_residual(
    params: typing.Tuple[float, float, float, float, float, float],
    grid: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
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


def fit_dipole(b_map, p0, x_grid, y_grid):
    """fits a single dipole to a magnetic field map

    Parameters
    ----------
    b_map: ndarray
        magnetic field map in Tesla
    p0: tuple(floats)
        initial guess for dipole parameters
    x_grid: ndarray
        x grid in meters
    y_grid: ndarray
        y grid in meters

    Returns
    -------
    dipole parameters: tuple(floats)
        dipole parameters [x_source, y_source, z_source, mx, my, mz]
    """
    grid = np.vstack((x_grid.ravel(), y_grid.ravel()))
    return least_squares(
        dipole_residual,
        p0,
        args=(grid, b_map.ravel()),
        loss="huber",
        method="trf",
        gtol=2.3e-16,
        ftol=2.3e-16,
        xtol=2.3e-16,
        max_nfev = 5000,
    )
