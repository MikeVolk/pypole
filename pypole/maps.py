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

calculate_map(x_grid: np.ndarray, y_grid: np.ndarray, locations: np.ndarray,
              source_vectors: np.ndarray, sensor_distance: float = 5e-6) -> np.ndarray:
    Calculate the magnetic field map for a set of sources.

Examples
--------
>>> x_grid, y_grid = get_grid(pixels=(50, 50), pixel_size=1e-6)
>>> locations, source_vectors = get_random_sources(n_sources=10)
>>> map_data = calculate_map(x_grid, y_grid, locations, source_vectors)


The maps.py file contains three functions for generating magnetic field maps from magnetic dipole sources.

- `get_grid()` generates a grid of observation coordinates for the map, given the number of pixels and pixel size.
- `get_random_sources()` generates a dictionary of random point source parameters including location and dipole moment, given the number of sources and range of parameters.
- `calculate_map()` calculates the magnetic field map for a set of sources, given the observation coordinates, source locations and dipole moments.

All functions return ndarrays of appropriate dimensions and data types.

Examples are included in the docstring to demonstrate the usage of each function.
"""

import typing
from typing import Any, Tuple, Union
from numba import njit
import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pypole import NDArray64, convert

LOG = logging.getLogger(__name__)


def get_random_sources(
    n_sources: int,
    x_range: tuple[float, float] = (-3.0e-6, 3.0e-6),
    y_range: tuple[float, float] = (-3.0e-6, 3.0e-6),
    z_range: tuple[float, float] = (1e-6, 4e-6),
    moment_range: tuple[float, float] = (1e-14, 1e-14),
) -> tuple[NDArray64, NDArray64]:
    """Generate diction of point source parameters

    Parameters
    ----------
    n_sources: int
             Number of sources per map
    x_range:  tuple (min, max)
        range for possible values
    y_range: tuple (min, max)
        range for possible values
    z_range: tuple (min, max)
        range of z for each source
    moment_range:  tuple (min, max)
        range of moments to generate

    Returns
    -------
        ndarrays of location and source_vector
            location (n_sources, 3): x_source, y_source, z_source
            source_vector (n_sources, 3): x,y,z components of dipole moment

    Notes
    -----
    The range of x,y positions for the dipole is currently limited to the central micron
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
    moment_range : tuple
        Range of moment magnitudes to generate in Am^2. Tuple format is (min, max).

    Returns
    -------
    dim : ndarray
        Dipole moments in (declination, inclination, moment) format. Declination and inclination are given in degrees,
        while moment is given in Am^2.

    Examples
    --------
    >>> np.random.seed(0)
    >>> get_random_dim(2, (1e-14, 1e-14))
    array([[ 1.97572861e+02, -1.18603363e+01,  1.00000000e-14],
           [ 2.57468172e+02, -5.15016644e+00,  1.00000000e-14]])

    Notes
    -----
    The dipole moments are generated using the following process:
        1. Generate random declinations from a uniform distribution between 0 and 360 degrees.
        2. Generate random inclinations from a uniform distribution between -90 and 90 degrees.
        3. Generate uniform distribution of moment magnitudes between the specified range.
        4. Combine declinations, inclinations, and moment magnitudes into a single numpy array in the order
           (declination, inclination, moment).

    The resulting array contains the dipole moments for each source in the map. The dipole moments are randomly
    distributed on the unit sphere, with the magnitude of each dipole moment randomly distributed between the
    specified range.
    """
    # get random declinations from uniform distribution
    declination = np.random.uniform(0, 360, n_sources)
    # get random inclinations from uniform distribution
    inclination = np.rad2deg(np.arccos(2 * np.random.rand(n_sources) - 1)) - 90
    # get uniform distribution of moment magnitude
    moment_scalar = np.random.uniform(moment_range[0], moment_range[1], size=n_sources)
    return np.stack([declination, inclination, moment_scalar]).T



def get_random_locations(n_sources, x_range, y_range, z_range):
    """Generate random locations for sources within the source region of the map (x_range, y_range, z_range)

    Parameters
    ----------
    n_sources: int
        number of sources per map
    x_range: tuple (min, max)
        range for possible values
    y_range: tuple (min, max)
        range for possible values
    z_range: tuple (min, max)
        range of z for each source

    Returns
    -------
    locations: ndarray
        x,y,z locations of sources

    Examples
    --------
    >>> np.random.seed(0)
    >>> get_random_locations(2, (-3e-6, 3e-6), (-3e-6, 3e-6), (1e-6, 4e-6))
    >>> array([[2.92881024e-07, 6.16580256e-07, 2.27096440e-06],[1.29113620e-06, 2.69299098e-07, 2.93768234e-06]])

    """

    # get uniform distribution of x/y locations
    x_source = np.random.uniform(x_range[0], x_range[1], size=n_sources)
    y_source = np.random.uniform(y_range[0], y_range[1], size=n_sources)
    z_source = np.random.uniform(z_range[0], z_range[1], size=n_sources)
    return np.stack([x_source, y_source, z_source]).T


def get_grid(
    pixels: Union[Tuple[int, int], int] = (100, 100), pixel_size: float = 5e-6
) -> Tuple[NDArray64, NDArray64]:
    """Generate observation coordinates of the map

    Parameters
    ----------
    pixels: Union[Tuple[int, int], int], optional
        Number of grid points (i.e. pixels) in the map. If an int is provided, a square map will be generated.
        Defaults to (100, 100).
    pixel_size: float, optional
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
    """
    if isinstance(pixels, int):
        LOG.warning(
            f"pixels should be a tuple of (x,y) pixel size. Setting to ({pixels},{pixels})"
        )
        pixels = (pixels, pixels)

    x_points = np.linspace(-pixels[0], pixels[0], pixels[0]) * pixel_size / 2
    y_points = np.linspace(-pixels[1], pixels[1], pixels[1]) * pixel_size / 2

    x_grid, y_grid = np.meshgrid(x_points, y_points)
    return x_grid.astype(np.float64), y_grid.astype(np.float64)

