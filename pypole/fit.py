"""
fit.py
------

Functions for fitting a single magnetic dipole to a field map.

The main function is `fit_dipole`, which takes a magnetic field map and an initial guess for the dipole
parameters, and returns the optimized dipole parameters. This function uses `_fit_dipole`, which is a
helper function that actually performs the optimization using the `scipy.optimize.least_squares`
function.

Functions
---------
fit_dipole(field_map, p0, pixel_size=1.0)
    Fit a single dipole to a magnetic field map using non-linear least squares optimization.

_fit_dipole(field_map, p0, x_grid, y_grid)
    Helper function for fitting a single dipole to a magnetic field map.

fit_dipole_n_maps(x_grid, y_grid, field_maps, initial_guess)
    Fit a series of maps, each with a single dipole.

__initial_guess_from_synthetic(mvec)
    Get initial guess for dipole parameter calculation.

dipole_residual(params, grid, data)
    Calculate the residual between the calculated magnetic field map and the input magnetic field map.

"""

import typing

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, OptimizeResult

from pypole import NDArray64, compute, maps
from pypole.dipole import dipole_field


@numba.njit(parallel=True)
def fit_dipole_n_maps(
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    field_maps: NDArray[np.float64],
    initial_guess: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Fits a series of maps, each with a single dipole.

    This function takes a set of magnetic field maps and initial guesses for the dipole parameters for each map,
    and returns an array of best fit dipole parameters for each map.

    Parameters
    ----------
    x_grid : ndarray (pixel, pixel)
        x grid
    y_grid : ndarray (pixel, pixel)
        y grid
    field_maps : ndarray (n_maps, pixel, pixel)
        Magnetic field maps for all frames.
    initial_guess : ndarray (n_maps, 3)
        Initial guesses for dipole parameters for each map.

    Returns
    -------
    best_fit_dipoles : ndarray (n_maps, 6)
        Array of best fit dipole parameters for each map.

    Notes
    -----
    This function uses Numba's JIT compiler to optimize performance by parallelizing the for loop.

    The input arrays `field_maps` and `initial_guess` should be of the same length (i.e., same number of maps).
    The size of the output `best_fit_dipoles` array will be equal to the number of maps in the input array.

    Example
    -------
    >>> import numpy as np
    >>> from pypole import maps, fit
    >>> np.random.seed(0)
    >>> n_maps = 3
    >>> n_sources = 1
    >>> location, moment = maps.get_random_sources(n_sources)
    >>> field_maps = np.array([maps.calculate_map(*maps.get_grid(), location, moment) for _ in range(n_maps)])
    >>> initial_guess = np.array([fit.__initial_guess_from_synthetic(moment[0]) for _ in range(n_maps)])
    >>> best_fit_dipoles = fit.fit_dipole_n_maps(*maps.get_grid(), field_maps, initial_guess)
    >>> print(best_fit_dipoles)
    [[ 3.10503547e-06  3.14291279e-06  2.52005549e-06  2.46976879e-16  7.70811479e-16 -2.11717939e-15]
     [ 3.10503547e-06  3.14291279e-06  2.52005549e-06  2.46976879e-16  7.70811479e-16 -2.11717939e-15]
     [ 3.10503547e-06  3.14291279e-06  2.52005549e-06  2.46976879e-16  7.70811479e-16 -2.11717939e-15]]

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
    Compute an initial guess for the dipole parameters from a given magnetic moment vector.

    Parameters
    ----------
    mvec : ndarray (3,)
        Magnetic moment vector in A/m^2.

    Returns
    -------
    initial_guess : ndarray (6,)
        Initial guess for dipole parameters in the format (x_source, y_source, z_source, mx, my, mz).
        The position components are set to zero and the moment components are taken directly from the input
        magnetic moment vector.

    Notes
    -----
    This function computes an initial guess for the dipole parameters to be used in the `fit_dipole` function.
    The `x_source`, `y_source`, and `z_source` components of the dipole are set to zero, while the `mx`, `my`,
    and `mz` components are taken directly from the input magnetic moment vector.

    The returned dipole parameters have units of meters for the position components and Am^2 for the moment
    components.

    Examples
    --------
    >>> mvec = np.array([1.0, 0.0, 0.0])
    >>> initial_guess = __initial_guess_from_synthetic(mvec)
    >>> print(initial_guess)
    [0. 0. 5e-6 1. 0. 0.]
    """
    return np.array([0, 0, 5e-6, mvec[0], mvec[1], mvec[2]])

def dipole_residual(
    params: tuple[float, float, float, float, float, float],
    grid: NDArray[np.float64],
    data: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the residual between the calculated magnetic field map and the input magnetic field map
    for fitting a single dipole.

    Args:
        params (tuple[float, float, float, float, float, float]): The parameters for field calculation:
            [x_source, y_source, z_source, mx, my, mz].
        grid (ndarray): The x and y grid of the magnetic field map.
        data (ndarray): The magnetic field data.

    Returns:
        ndarray: The difference between the calculated magnetic field map from params and the input
        magnetic field map.

    Raises:
        None

    Notes:
        This function calculates the difference between the calculated magnetic field map and the input
        magnetic field map for fitting a single magnetic dipole. The function takes dipole parameters, x and y
        grid, and magnetic field data as inputs, and returns the difference between the calculated magnetic
        field map and the input magnetic field map. This difference is calculated by subtracting the calculated
        magnetic field map from the input magnetic field map.

    Examples:
        >>> x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        >>> B = pypole.maps.synthetic_map(x, y, np.array([0]), np.array([0]), np.array([0.01]), np.array([0]), np.array([0]), np.array([1]))
        >>> params = (0, 0, 0.01, 0, 0, 1)
        >>> dipole_residual(params, (x, y), B)
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
    """
    Fit a single dipole to a magnetic field map using non-linear least squares optimization.

    Parameters
    ----------
    field_map : ndarray (pixel, pixel)
        The magnetic field map in units of Tesla.
    p0 : tuple of floats
        The initial guess for dipole parameters in the format (x_source, y_source, z_source, mx, my, mz).
        `x_source`, `y_source`, and `z_source` are the coordinates of the dipole source in meters.
        `mx`, `my`, and `mz` are the dipole moment vector components in units of Ampere-meter^2 (Am^2).
    pixel_size : float, optional
        The size of each pixel in the magnetic field map in meters. Default value is 1.0.

    Returns
    -------
    dipole_parameters : tuple of floats
        The dipole parameters in the format (x_source, y_source, z_source, mx, my, mz).

    Raises
    ------
    ValueError
        If the length of `p0` is not equal to 6.

    Notes
    -----
    This function uses least squares optimization to fit a single dipole to the given
    magnetic field map. The optimization is performed using the `dipole_residual` function,
    which computes the residual error between the model field of a single dipole and
    the given field map. The optimization algorithm used is 'trf' (Trust Region Reflective),
    which works well for a large range of problem types. The loss function used is 'huber',
    which is less sensitive to outliers in the data compared to 'linear' loss.

    Examples
    --------
    >>> import pypole.maps
    >>> import numpy as np
    >>> from pypole import fit
    >>> np.random.seed(0)
    >>> n_sources = 1
    >>> location, moment = pypole.maps.get_random_sources(n_sources)
    >>> B = pypole.maps.calculate_map(*pypole.maps.get_grid(), location, moment)
    >>> p = (*location[0], *moment[0])
    >>> dipole_fit = fit.fit_dipole(B, p)
    >>> print(dipole_fit)
    (3.105035473030962e-06, 3.1429127897760807e-06, 2.520055491162705e-06, 2.469768791414657e-16, 7.708114785111617e-16, -2.1171793943168296e-15)

    """


    x_grid, y_grid = maps.get_grid(pixels=field_map.shape, pixel_size=pixel_size)
    return _fit_dipole(field_map, p0, x_grid, y_grid)


def _fit_dipole(
    field_map: NDArray64,
    p0: tuple[float, float, float, float, float, float],
    x_grid: NDArray64,
    y_grid: NDArray64,
) -> OptimizeResult:
    """Helper function for fitting a single dipole to a magnetic field map.

    Args:
        field_map: A 2D numpy array of the magnetic field map in Tesla.
        p0: A tuple of 6 floats representing the initial guess for dipole parameters [x_source, y_source, z_source, mx, my, mz].
        x_grid: A 2D numpy array of x-coordinates of the magnetic field map.
        y_grid: A 2D numpy array of y-coordinates of the magnetic field map.

    Returns:
        An OptimizeResult object containing the results of the least-squares optimization.

    Notes:
        This function is a helper function for `fit_dipole` and should not be called directly. It uses the least-squares
        optimization algorithm to fit a single dipole to the given magnetic field map. The optimization is performed
        using the `dipole_residual` function, which computes the residual error between the model field of a single
        dipole and the given field map. The optimization algorithm used is 'trf' (Trust Region Reflective), which
        works well for a large range of problem types. The loss function used is 'huber', which is less sensitive to
        outliers in the data compared to 'linear' loss.
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
