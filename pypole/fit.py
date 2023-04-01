"""
Functions for fitting a single magnetic dipole to a field map.

The main function is `fit_dipole`, which takes a magnetic field map and an initial guess for the dipole
parameters, and returns the optimized dipole parameters. This function uses `_fit_dipole`, which is a
helper function that actually performs the optimization using the `scipy.optimize.least_squares`
function.

Functions
---------
fit_dipole(field_map: ndarray, p0: Tuple[float, float, float, float, float, float], pixel_size: float = 1) -> Tuple[float, float, float, float, float, float]:
    fits a single dipole to a magnetic field map

_fit_dipole(field_map: ndarray, p0: Tuple[float, float, float, float, float, float], x_grid: ndarray, y_grid: ndarray) -> OptimizeResult:
    helper function that fits a single dipole to a magnetic field map using `scipy.optimize.least_squares`
"""

import typing

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from pypole import NDArray64, compute, maps
from pypole.dipole import dipole_field


@jit(nopython=True, parallel=True)
def fit_dipole_n_maps(
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    field_maps: NDArray[np.float64],
    initial_guess: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit a series of maps, each with a single dipole.

    Parameters
    ----------
    x_grid : ndarray(pixel, pixel)
        x grid
    y_grid : ndarray(pixel, pixel)
        y grid
    field_maps : ndarray(n_maps, pixel, pixel)
        magnetic field map for all frames
    initial_guess : ndarray(n_maps, 3)
        initial guess for dipole parameters

    Returns
    -------
    best_fit_dipoles : ndarray(n_maps, 6)
        array of best fit dipole parameters for each map

    Notes
    -----
    This function uses Numba's JIT compiler to optimize performance by parallelizing the for loop.

    """

    n_maps = field_maps.shape[0]
    best_fit_dipoles = np.empty((n_maps, 6))

    for map_index in numba.prange(n_maps):
        # fit dipole for each map
        best_fit_dipoles[map_index, :] = _fit_dipole(
            field_map=field_maps[map_index],
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
    grid: NDArray[np.float64],
    data: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the residual between the calculated magnetic field map
    and the input magnetic field map for fitting a single dipole.

    Parameters
    ----------
    params : tuple of floats
        The parameters for field calculation: [x_source, y_source, z_source, mx, my, mz].
    grid : ndarray
        The x and y grid of the magnetic field map.
    data : ndarray
        The magnetic field data.

    Returns
    -------
    residuals : ndarray
        The difference between the calculated magnetic field map from params and the
        input magnetic field map.
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


def fit_dipole(
    field_map: NDArray64,
    p0: tuple[float, float, float, float, float, float],
    pixel_size: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    """Fit a single dipole to a magnetic field map using non-linear least squares optimization.

    Parameters
    ----------
    field_map : ndarray (pixel, pixel)
        Magnetic field map in Tesla
    p0 : tuple of floats
        Initial guess for dipole parameters: (x_source, y_source, z_source, mx, my, mz)
    pixel_size : float, optional
        Pixel size in meters, default is 1.0

    Returns
    -------
    dipole_parameters : tuple of floats
        Dipole parameters in the format (x_source, y_source, z_source, mx, my, mz)

    Raises
    ------
    ValueError
        If the length of `p0` is not equal to 6.

    Notes
    -----
    This function uses least squares optimization to fit a single dipole to the given
    magnetic field map. The optimization is performed using the dipole_residual function,
    which computes the residual error between the model field of a single dipole and
    the given field map. The optimization algorithm used is 'trf' (Trust Region Reflective),
    which works well for a large range of problem types. The loss function used is 'huber',
    which is less sensitive to outliers in the data compared to 'linear' loss.

    Examples
    --------
    >>> import numpy as np
    >>> from pypole import fit, maps
    >>> np.random.seed(0)
    >>> n_sources = 1
    >>> location, moment = maps.get_random_sources(n_sources)
    >>> B = maps.calculate_map(*maps.get_grid(), location, moment)
    >>> p = (*location[0], *moment[0])
    >>> dipole_fit = fit.fit_dipole(B, p)
    >>> print(dipole_fit)
    (3.105035473030962e-06, 3.1429127897760807e-06, 2.520055491162705e-06, 2.469768791414657e-16, 7.708114785111617e-16, -2.1171793943168296e-15)
    """

    x_grid, y_grid = maps.get_grid(pixels=field_map.shape, pixel_size=pixel_size)
    return _fit_dipole(field_map, p0, x_grid, y_grid)


def _fit_dipole(field_map: NDArray64, p0: tuple[float, float, float, float, float, float], x_grid: NDArray64, y_grid: NDArray64) -> OptimizeResult:
    """Helper function for fitting a single dipole to a magnetic field map.

    Parameters
    ----------
    field_map : ndarray
        Magnetic field map in Tesla.
    p0 : tuple(floats)
        Initial guess for dipole parameters [x_source, y_source, z_source, mx, my, mz].
    x_grid : ndarray
        X grid for magnetic field map.
    y_grid : ndarray
        Y grid for magnetic field map.

    Returns
    -------
    OptimizeResult
        Result of least squares optimization.

    Notes
    -----
    This function is a helper function for fit_dipole and should not be called directly.
    """
    grid = np.vstack((y_grid.ravel(), x_grid.ravel()))
    return least_squares(
        dipole_residual,
        p0,
        args=(grid, field_map.T.ravel()),
        loss="huber",
        method="trf",
        gtol=2.3e-16,
        ftol=2.3e-16,
        xtol=2.3e-16,
        max_nfev=5000,
    )
