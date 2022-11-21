from typing import Any

import itertools

import numba
import numpy as np
from numpy.typing import NDArray

from pypole import NDArray64
from pypole.maps import get_grid, get_random_sources


def synthetic_map(
    n_sources: int = 100,
    pixels: int = 100,
    sensor_distance: float = 5e-6,
    pixel_size: float = 1e-6,
) -> NDArray64:
    """Calculate a single simple magnetic field map for a number of sources.

    Parameters
    ----------
    n_sources : int
        number of sources
    pixels : int
        number of pixels in the map
    sensor_distance : float
        distance from the sensor to the sample
    pixel_size : float
        size of each pixel in meters

    Returns
    -------
    b_map : ndarray(pixels, pixels)
        magnetic field map
    """
    # get the map grid in meters
    x_grid, y_grid = get_grid(pixels, pixel_size)

    # get the location for the sources
    locations, source_vectors = get_random_sources(n_sources)
    return calculate_map(x_grid, y_grid, locations, source_vectors, sensor_distance)


def calculate_map(
    x_grid: NDArray64,
    y_grid: NDArray64,
    locations: NDArray64,
    source_vectors: NDArray64,
    sensor_distance: float = 5e-6,
) -> NDArray[np.float_]:
    """Calculate the magnetic field map for a set of sources

    Parameters
    ----------
    x_grid : ndarray(pixel, pixel)
        x grid
    y_grid : ndarray(pixel, pixel)
        y grid
    locations : ndarray(n_sources, 3)
        x,y,z locations of sources
    source_vectors : ndarray(n_sources, 3)
        dipole moments in (x, y, z) format and Am2
    sensor_distance : float, optional
        sensor-sample distance by default 0.0
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


@numba.jit(fastmath=True)
def dipole_field(
    x_grid: NDArray64,
    y_grid: NDArray64,
    x_source: NDArray64,
    y_source: NDArray64,
    z_observed: NDArray64,
    mx: float,
    my: float,
    mz: float,
) -> NDArray64:
    """Compute the field of a magnetic dipole point source

    Parameters
    ----------
    x_source: ndarray (n_sources, 1)
        x-locations of source
    y_source:  ndarray (n_sources, 1)
        y-locations of source
    z_observed: ndarray(n_sources)
        observed z distance, including the sensor height
    x_grid:  ndarray(pixel, pixel)
        grid to calculate the fields for
    y_grid: ndarray(pixel, pixel)
        grid to calculate the fields on
    mx: ndarray(n_sources,)
        x-component of vector
    my: ndarray(n_sources,)
        y-component of vector
    mz: ndarray(n_sources,)
        z-component of vector
    """
    dgridx = np.subtract(x_grid, x_source)
    dgridy = np.subtract(y_grid, y_source)

    squared_distance = np.square(dgridx) + np.square(dgridy) + np.square(z_observed)

    aux = mx * dgridx + my * dgridy + mz * z_observed
    aux /= np.sqrt(np.power(squared_distance, 5.0))
    return 1e-7 * (
        3.0 * aux * z_observed - mz / np.sqrt(np.power(squared_distance, 3.0))
    )
