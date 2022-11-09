import typing
from typing import Tuple, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pypole import convert


def get_random_sources(
    n_sources: int,
    x_range: tuple = (-3.0e-6, 3.0e-6),
    y_range: tuple = (-3.0e-6, 3.0e-6),
    z_range: tuple = (1e-6, 4e-6),
    moment_range: tuple = (1e-14, 1e-14),
) -> tuple[NDArray, NDArray]:
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
        ndarrays of location, source_vector, and total_vector
            location: x_source, y_source, z_source
            source_vector: x,y,z
            total_vector: x,y,z

    Notes
    -----
    The range of x,y positions for the dipole is currently limited to the central micron'
    """

    # get the locations of the sources
    locations = get_random_locations(n_sources, x_range, y_range, z_range)
    # reshape total location to be (n_maps, n_sources, (x,y,z))
    locations = locations.reshape(n_sources, 3)

    # calculate the x/y/z components
    dim = get_random_dim(moment_range, n_sources)
    xyz = convert.dim2xyz(dim)

    # reshape xyz to be (n_sources, (x,y,z))
    source_vector = xyz.reshape(n_sources, 3)

    return locations, source_vector


def get_random_dim(moment_range, n_sources):
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
    """

    # get uniform distribution of x/y locations
    x_source = np.random.uniform(x_range[0], x_range[1], size=n_sources)
    y_source = np.random.uniform(y_range[0], y_range[1], size=n_sources)
    z_source = np.random.uniform(z_range[0], z_range[1], size=n_sources)
    return np.stack([x_source, y_source, z_source]).T


def get_grid(n_pixel: int = 100, pixel_size: float = 5e-6) -> NDArray:
    """Generate observation coordinates of the map

    Parameters
    ----------
    n_pixel: int
        number of grid points (i.e. pixel) in the map
    pixel_size: float
        size of a single pixel (x/y are the same) in micron.
        defines: left map edge = -(pixel*pixel_size)/2, right map edge = (pixel*pixel_size)/2
    """
    n_x_points = np.linspace(
        -(n_pixel * pixel_size) / 2, (n_pixel * pixel_size) / 2, n_pixel
    )
    n_y_points = np.linspace(
        -(n_pixel * pixel_size) / 2, (n_pixel * pixel_size) / 2, n_pixel
    )

    xgrid = np.ones((n_pixel, n_pixel)) * n_x_points
    ygrid = np.ones((n_pixel, n_pixel)) * n_y_points
    return xgrid, ygrid.T
