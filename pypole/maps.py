import typing
from typing import Any, Tuple, Union

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
    The range of x,y positions for the dipole is currently limited to the central micron'
    """

    # get the locations of the sources
    locations = get_random_locations(n_sources, x_range, y_range, z_range)
    # reshape total location to be (n_maps, n_sources, (x,y,z))
    locations = locations.reshape(n_sources, 3)

    # calculate the x/y/z components
    dim = get_random_dim(n_sources, moment_range)
    xyz = convert.dim2xyz(dim)

    # reshape xyz to be (n_sources, (x,y,z))
    source_vector = xyz.reshape(n_sources, 3)

    return locations, source_vector


def get_random_dim(n_sources, moment_range):
    """Generate randomly distributed dipole moments on the unit sphere

    Parameters
    ----------
    moment_range: tuple (min, max)
        range of moments to generate
    n_maps: int
        number of maps to generate
    n_sources: int
        number of sources per map

    Returns
    -------
    dim: ndarray
        dipole moments in (declination, inclination, moment) format

    Examples
    --------
    >>> np.random.seed(0)
    >>> get_random_dim(2,(1e-14, 1e-14))
    >>> array([[ 1.97572861e+02, -1.18603363e+01,  1.00000000e-14], [ 2.57468172e+02, -5.15016644e+00,  1.00000000e-14]])

    """
    # get random declinations from uniform distribution (rand)
    declination = np.random.uniform(0, 360, n_sources)
    # get random inclinations from uniform distribution (rand)
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
    pixels: Union[tuple[int, int], int] = (100, 100), pixel_size: float = 5e-6
) -> tuple[NDArray64, NDArray64]:
    """Generate observation coordinates of the map

    Parameters
    ----------
    n_pixel: int
        number of grid points (i.e. pixel) in the map
    pixel_size: float
        size of a single pixel (x/y are the same) in micron.
        defines: left map edge = -(pixel*pixel_size)/2, right map edge = (pixel*pixel_size)/2
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
