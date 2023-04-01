from typing import Any, Tuple

import itertools

import numba
import numpy as np
from numpy.typing import NDArray

from pypole import NDArray64
from pypole.maps import get_grid, get_random_sources


def synthetic_map(
    n_sources: int = 100,
    pixels: tuple[int, int] = (100, 100),
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
    field_map : ndarray(pixels, pixels)
        magnetic field map
    """
    # get the map grid in meters
    x_grid, y_grid = get_grid(pixels, pixel_size)

    # get the location for the sources
    locations, source_vectors = get_random_sources(n_sources)
    return calculate_map(x_grid, y_grid, locations, source_vectors, sensor_distance)


from numba import prange

@numba.jit(parallel=True, fastmath=True)
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



@numba.njit(fastmath=True, parallel=True)
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
    """
    Computes the magnetic field of a magnetic dipole point source.

    Parameters:
    -----------
    x_grid: ndarray, shape (pixel, pixel)
        The x-coordinates of the grid to calculate the fields for.
    y_grid: ndarray, shape (pixel, pixel)
        The y-coordinates of the grid to calculate the fields for.
    x_source: ndarray, shape (n_sources, )
        The x-locations of the magnetic dipoles.
    y_source: ndarray, shape (n_sources, )
        The y-locations of the magnetic dipoles.
    z_observed: ndarray, shape (n_sources, )
        The z-coordinates of the sensor positions. The sensor height is included in this
        value.
    mx: ndarray, shape (n_sources, )
        The x-component of the magnetic dipole moment vector in Am^2.
    my: ndarray, shape (n_sources, )
        The y-component of the magnetic dipole moment vector in Am^2.
    mz: ndarray, shape (n_sources, )
        The z-component of the magnetic dipole moment vector in Am^2.

    Returns:
    --------
    ndarray, shape (pixel, pixel)
        The magnetic field values at each (x, y) point on the grid.

Returns
    -------
    NDArray64
        The magnetic field generated by a dipole at each point on the grid.

    Notes
    -----
    This function computes the magnetic field generated by a magnetic dipole point source.
    The field is computed at each point on a 2D grid.

    The formula used in this function is based on the dipole equation for magnetic field,
    given by:

    B = μ₀/4π * [(3(r * m)r - m(r.r)) / r^5]

    where
    B: Magnetic field
    μ₀: Permeability of free space
    r: Position vector
    m: Magnetic moment
    r.m: Dot product of position vector and magnetic moment

    The formula has been modified to compute the field for a point source with a fixed
    position and magnetic moment.

    The units of input parameters are as follows:
    - x_grid, y_grid, x_source, y_source: meters
    - z_observed: meters above ground level
    - mx, my, mz: Am^2

    The unit of the output is Tesla (T).
    """
    # Calculate the distances between the grid points and the magnetic dipole sources
    dgridx = x_grid - x_source
    dgridy = y_grid - y_source

    # Calculate the squared distance between the grid points and the magnetic dipole sources
    squared_distance = np.square(dgridx) + np.square(dgridy) + np.square(z_observed)

    # Calculate the dot product of the magnetic dipole moment vector and the distance vector
    dot_product = mx * dgridx + my * dgridy + mz * z_observed

    # Calculate the field due to the x and y components of the magnetic dipole moment vector
    aux = dot_product / np.power(squared_distance, 5.0 / 2.0)
    # bx = 3.0 * aux * dgridx
    # by = 3.0 * aux * dgridy

    # Calculate the field due to the z component of the magnetic dipole moment vector
    bz = (3.0 * aux) * z_observed

    # Combine the x, y, and z components of the field to get the total field
    return 1e-7*bz
