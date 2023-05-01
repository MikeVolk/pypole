"""
maps.py
=======
This module contains functions for generating magnetic field maps from magnetic dipole sources.

Functions
---------
get_grid(pixels: Tuple[int, int] or int = (100, 100), pixel_size: float = 5e-6) -> Tuple[np.ndarray, np.ndarray]:
    Generate a grid of observation coordinates for the map.

get_random_sources(n_sources: int, x_range: Tuple[float, float] = (-3e-6, 3e-6),
                    y_range: Tuple[float, float] = (-3e-6, 3e-6),
                    z_range: Tuple[float, float] = (1e-6, 4e-6),
                    moment_range: Tuple[float, float] = (1e-14, 1e-14)) -> Tuple[np.ndarray, np.ndarray]:
    Generate a dictionary of random point source parameters including location and dipole moment.

get_random_dim(n_sources: int, moment_range: Tuple[float, float]) -> np.ndarray:
    Generate randomly distributed dipole moments on the unit sphere.

get_random_locations(n_sources: int, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float]) -> np.ndarray:
    Generate random locations for sources within the source region of the map (x_range, y_range, z_range)

calculate_map(x_grid: np.ndarray, y_grid: np.ndarray, locations: np.ndarray,
              source_vectors: np.ndarray, sensor_distance: float = 5e-6) -> np.ndarray:
    Calculate the magnetic field map for a set of sources.

synthetic_map(n_sources: int = 100,
              pixels: Tuple[int, int] = (100, 100),
              sensor_distance: float = 5e-6,
              pixel_size: float = 1e-6) -> np.ndarray:
    Calculate a single simple magnetic field map for a number of sources.

Examples
--------
>>> x_grid, y_grid = get_grid(pixels=(50, 50), pixel_size=1e-6)
>>> locations, source_vectors = get_random_sources(n_sources=10)
>>> map_data = calculate_map(x_grid, y_grid, locations, source_vectors)
>>> random_dim = get_random_dim(n_sources=10, moment_range=(1e-14, 1e-14))
>>> random_locations = get_random_locations(n_sources=10, x_range=(-3e-6, 3e-6), y_range=(-3e-6, 3e-6), z_range=(1e-6, 4e-6))

The `maps.py` file contains several functions for generating magnetic field maps from magnetic dipole sources.

- `get_grid()` generates a grid of observation coordinates for the map, given the number of pixels and pixel size.
- `get_random_sources()` generates a dictionary of random point source parameters including location and dipole moment, given the number of sources and range of parameters.
- `get_random_dim()` generates randomly distributed dipole moments on the unit sphere.
- `get_random_locations()` generates random locations for sources within the source region of the map (x_range, y_range, z_range).
- `calculate_map()` calculates the magnetic field map for a set of sources, given the observation coordinates, source locations, and dipole moments.
- `synthetic_map()` calculates a single simple magnetic field map for a number of sources.

All functions return ndarrays of appropriate dimensions and data types

"""

import typing
from typing import Any, Tuple, Union

import numba
from numba import njit
import logging

import numpy as np
from numpy._typing import NDArray
from numpy.typing import ArrayLike, NDArray

from pypole import NDArray64, convert
from pypole.dipole import dipole_field

LOG = logging.getLogger(__name__)


def get_random_sources(
    n_sources: int,
    x_range: tuple[float, float] = (-3.0e-6, 3.0e-6),
    y_range: tuple[float, float] = (-3.0e-6, 3.0e-6),
    z_range: tuple[float, float] = (1e-6, 4e-6),
    moment_range: tuple[float, float] = (1e-14, 1e-14),
) -> tuple[NDArray64, NDArray64]:
    """
    Generates random magnetic dipole sources with location and dipole moment parameters.

    Parameters
    ----------
    n_sources : int
        Number of magnetic dipole sources to generate.
    x_range : Tuple[float, float], optional
        Range of x values for the source locations, in meters. Default is (-3.0e-6, 3.0e-6).
    y_range : Tuple[float, float], optional
        Range of y values for the source locations, in meters. Default is (-3.0e-6, 3.0e-6).
    z_range : Tuple[float, float], optional
        Range of z values for the source locations, in meters. Default is (1e-6, 4e-6).
    moment_range : Tuple[float, float], optional
        Range of dipole moment magnitudes for the sources, in Am^2. Default is (1e-14, 1e-14).

    Returns
    -------
    Tuple[NDArray64, NDArray64]
        A tuple containing the location and dipole moment vectors for each source.
        - location : ndarray(n_sources, 3)
            A 2D array of shape (n_sources, 3) containing the (x, y, z) coordinates of each source.
        - dipole_moment : ndarray(n_sources, 3)
            A 2D array of shape (n_sources, 3) containing the (x, y, z) components of the dipole moment vector
            for each source, with magnitude randomly distributed within the specified range.

    Notes
    -----
    This function generates `n_sources` random magnetic dipole sources with randomly distributed location and dipole moment parameters within the specified ranges. The dipole moments are generated on the unit sphere, with the magnitude of each dipole moment randomly distributed between the specified `moment_range`. The location of each source is randomly generated within the specified `x_range`, `y_range`, and `z_range`.

    Example
    -------
    Generate 10 random magnetic dipole sources within a square of side length 6 microns and with a dipole moment magnitude of 1e-14 Am^2:

    >>> locations, dipole_moments = get_random_sources(n_sources=10, x_range=(-3e-6, 3e-6), y_range=(-3e-6, 3e-6), z_range=(1e-6, 4e-6), moment_range=(1e-14, 1e-14))
    """

    # get the locations of the sources
    x_source = np.random.uniform(x_range[0], x_range[1], size=n_sources)
    y_source = np.random.uniform(y_range[0], y_range[1], size=n_sources)
    z_source = np.random.uniform(z_range[0], z_range[1], size=n_sources)
    locations = np.stack([x_source, y_source, z_source]).T

    # calculate the x/y/z components
    declination = np.random.uniform(0, 360, n_sources)
    inclination = np.rad2deg(np.arccos(2 * np.random.rand(n_sources) - 1)) - 90
    moment_scalar = np.random.uniform(moment_range[0], moment_range[1], size=n_sources)
    dim = np.stack([declination, inclination, moment_scalar]).T
    source_vector = np.zeros((n_sources, 3))
    convert.dim2xyz(dim, source_vector)

    return locations, source_vector


def get_random_dim(n_sources: int, moment_range: Tuple[float, float]) -> NDArray64:
    """
    Generate randomly distributed dipole moments on the unit sphere.

    Parameters
    ----------
    n_sources : int
        Number of sources per map.
    moment_range : tuple of float
        Tuple of minimum and maximum moment magnitudes in Am^2 to generate.

    Returns
    -------
    dim : ndarray
        Dipole moments in (declination, inclination, moment) format. Declination and inclination are given in degrees,
        while moment is given in Am^2.

    Notes
    -----
    This function generates random dipole moments on the unit sphere in three steps:
        1. Generate n_sources random declinations between 0 and 360 degrees.
        2. Generate n_sources random inclinations between -90 and 90 degrees.
        3. Generate n_sources random moment magnitudes between moment_range[0] and moment_range[1] Am^2.
    The resulting dipole moments are returned as a ndarray of shape (n_sources, 3), where each row corresponds to a
    single dipole moment in the (declination, inclination, moment) format.

    Examples
    --------
    >>> np.random.seed(0)
    >>> get_random_dim(2, (1e-14, 1e-14))
    array([[ 1.97572861e+02, -1.18603363e+01,  1.00000000e-14],
           [ 2.57468172e+02, -5.15016644e+00,  1.00000000e-14]])

    """

    # get random declinations from uniform distribution
    declination = np.random.uniform(0, 360, n_sources)
    # get random inclinations from uniform distribution
    inclination = np.rad2deg(np.arccos(2 * np.random.rand(n_sources) - 1)) - 90
    # get uniform distribution of moment magnitude
    moment_scalar = np.random.uniform(moment_range[0], moment_range[1], size=n_sources)
    return np.stack([declination, inclination, moment_scalar]).T


def get_random_locations(n_sources, x_range, y_range, z_range):
    """
    Generate random locations for sources within the specified x, y, and z ranges.

    Parameters
    ----------
    n_sources : int
        The number of sources to generate locations for.
    x_range : tuple of floats
        A tuple (min, max) representing the minimum and maximum x-values for the source locations.
    y_range : tuple of floats
        A tuple (min, max) representing the minimum and maximum y-values for the source locations.
    z_range : tuple of floats
        A tuple (min, max) representing the minimum and maximum z-values for the source locations.

    Returns
    -------
    np.ndarray
        An array of shape (n_sources, 3) containing the x, y, and z coordinates for the randomly generated sources.

    Examples
    --------
    >>> np.random.seed(0)
    >>> get_random_locations(2, (-3e-6, 3e-6), (-3e-6, 3e-6), (1e-6, 4e-6))
    array([[2.92881024e-07, 6.16580256e-07, 2.27096440e-06],
           [1.29113620e-06, 2.69299098e-07, 2.93768234e-06]])

    Notes
    -----
    This function generates random locations for sources within the specified x, y, and z ranges. The number of
    sources generated is given by `n_sources`. The x, y, and z ranges are specified using the `x_range`, `y_range`,
    and `z_range` parameters, respectively, which should be tuples of the form (min, max).

    The function returns an array of shape (n_sources, 3) containing the randomly generated x, y, and z coordinates
    for the sources. The x and y coordinates are randomly generated using a uniform distribution over the specified
    ranges, while the z coordinate is randomly generated using a uniform distribution over the z_range parameter.

    The generated locations can be used as input to the `calculate_map` function to calculate a magnetic field map for
    the sources.
    """

    # get uniform distribution of x/y locations
    x_source = np.random.uniform(x_range[0], x_range[1], size=n_sources)
    y_source = np.random.uniform(y_range[0], y_range[1], size=n_sources)
    z_source = np.random.uniform(z_range[0], z_range[1], size=n_sources)
    return np.stack([x_source, y_source, z_source]).T


def get_grid(
    pixels: Union[Tuple[int, int], int] = (100, 100), pixel_size: float = 5e-6
) -> Tuple[NDArray64, NDArray64]:
    """
    Generate a grid of observation coordinates for the map.

    Parameters
    ----------
    pixels : Union[Tuple[int, int], int], optional
        Number of grid points (i.e. pixels) in the map. If an int is provided, a square map will be generated.
        Defaults to (100, 100).
    pixel_size : float, optional
        Size of a single pixel in meters. Defines the size of the map in x and y directions.
        Defaults to 5e-6.

    Returns
    -------
    Tuple[NDArray64, NDArray64]
        The x and y coordinates of the generated grid.

    Examples
    --------
    >>> x, y = get_grid((50, 100), pixel_size=1e-5)
    >>> x.shape
    (100, 50)
    >>> y.shape
    (100, 50)

    Notes
    -----
    This function generates a grid of observation coordinates for the magnetic field map. The grid is defined by
    the number of pixels in the x and y directions, and the size of each pixel in meters. The coordinates are
    centered on the origin, and span an area defined by the product of the number of pixels and pixel size in each
    direction.

    If a single integer is provided for the `pixels` argument, a square grid with that many pixels in each
    direction will be generated. If a tuple of integers is provided, a rectangular grid with the specified
    number of pixels in each direction will be generated.

    The function returns two ndarrays, `x` and `y`, representing the coordinates of the generated grid. These
    arrays can be used as input to the `calculate_map()` function to generate a magnetic field map.
    """
    if isinstance(pixels, int):
        LOG.warning(f"pixels should be a tuple of (x,y) pixel size. Setting to ({pixels},{pixels})")
        pixels = (pixels, pixels)

    x_points = np.linspace(-pixels[0], pixels[0], pixels[0]) * pixel_size / 2
    y_points = np.linspace(-pixels[1], pixels[1], pixels[1]) * pixel_size / 2

    x_grid, y_grid = np.meshgrid(x_points, y_points)
    return x_grid.astype(np.float64), y_grid.astype(np.float64)


def synthetic_map(
    n_sources: int = 100,
    pixels: tuple[int, int] = (100, 100),
    sensor_distance: float = 5e-6,
    pixel_size: float = 1e-6,
) -> NDArray64:
    """
    Calculate a synthetic magnetic field map for a set of randomly generated magnetic dipoles.

    Parameters
    ----------
    n_sources : int, optional
        The number of magnetic dipoles to include in the map. Default is 100.
    pixels : Tuple[int, int], optional
        The number of pixels in the x and y directions of the map, specified as a tuple. Default is (100, 100).
    sensor_distance : float, optional
        The distance between the magnetic sensor and the sample in meters. Default is 5e-6.
    pixel_size : float, optional
        The size of each pixel in meters. Default is 1e-6.

    Returns
    -------
    field_map : NDArray64
        The resulting magnetic field map as a 2D numpy array of shape (pixels[0], pixels[1]).

    Examples
    --------
    >>> map_data = synthetic_map(n_sources=10, pixels=(50, 50), sensor_distance=1e-6, pixel_size=1e-6)
    """
    # get the map grid in meters
    x_grid, y_grid = get_grid(pixels, pixel_size)

    # get the location for the sources
    locations, source_vectors = get_random_sources(n_sources)
    return calculate_map(x_grid, y_grid, locations, source_vectors, sensor_distance)


@numba.jit(parallel=True, fastmath=True)
def calculate_map(
    x_grid: NDArray64,
    y_grid: NDArray64,
    locations: NDArray64,
    source_vectors: NDArray64,
    sensor_distance: float = 5e-6,
) -> NDArray[np.float_]:
    """Calculate the magnetic field map for a set of magnetic dipole sources.

    Parameters
    ----------
    x_grid : ndarray, shape (n_pixels_x, n_pixels_y)
        The x-coordinates of the observation grid for the map.
    y_grid : ndarray, shape (n_pixels_x, n_pixels_y)
        The y-coordinates of the observation grid for the map.
    locations : ndarray, shape (n_sources, 3)
        The (x, y, z) coordinates of the magnetic dipole sources in meters.
    source_vectors : ndarray, shape (n_sources, 3)
        The (x, y, z) components of the dipole moment vectors of the sources in Am^2.
    sensor_distance : float, optional
        The distance between the sensor and the sample in meters. Default is 5e-6.

    Returns
    -------
    field_map : ndarray, shape (n_pixels_x, n_pixels_y)
        The magnetic field map calculated from the sources and observation grid.

    Notes
    -----
    The magnetic field at each point in the observation grid is calculated as the sum of the magnetic field
    contributions from each source. The magnetic field due to a single magnetic dipole source at a point (x, y, z)
    in space is given by:

    Bx = (mu_0 / 4*pi) * (3 * dx * r - x * r**3) / r**5
    By = (mu_0 / 4*pi) * (3 * dy * r - y * r**3) / r**5
    Bz = (mu_0 / 4*pi) * (3 * dz * r - z * r**3) / r**5

    where:
        - mu_0 is the magnetic constant (4 * pi * 1e-7 T m/A)
        - dx, dy, dz are the components of the dipole moment vector
        - r is the distance between the point and the source

    This function uses the implementation of the above equation in the `dipole_field()` function of the `pypole` library,
    which is a numba-accelerated implementation of the Biot-Savart law.

    The `numba.jit()` decorator is used to speed up the calculation by compiling the function to native machine code.
    The `parallel` and `fastmath` options of the decorator enable parallelization and faster math operations,
    respectively.
    """

    n_sources: int = locations.shape[0]
    b: NDArray64 = np.zeros((n_sources, x_grid.shape[0], x_grid.shape[1]))

    for i in range(n_sources):
        b[i, :, :] = dipole_field(
            x_grid,
            y_grid,
            locations[i, 0],
            locations[i, 1],
            locations[i, 2] + sensor_distance,
            source_vectors[i, 0],
            source_vectors[i, 1],
            source_vectors[i, 1],
        )
    return np.sum(b, axis=0)
