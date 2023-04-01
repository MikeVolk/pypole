import logging

import numba
import numpy as np
from numba import float64, guvectorize, int64
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)

from pypole import NDArray64

_EPSILON = 1e-50


def dipolarity_param(field_map: NDArray64, fitted_map: NDArray64) -> np.float64:
    """
    Calculate the dipolarity parameter of a magnetic dipole field.

    The dipolarity parameter (DP) is defined as the ratio of the rms of the
    residual (fitted_map - map) to the rms of the data map.
    DP was first introduced by [1]_.

    Args:
        field_map (NDArray64): Original magnetic field map.
        fitted_map (NDArray64): Fitted magnetic field map.

    Returns:
        np.float64: Dipolarity parameter of the magnetic dipole field.

    Raises:
        ValueError: If the shapes of the input arrays do not match.

    Examples:
        >>> import numpy as np
        >>> field_map = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> fitted_map = np.array([[1.2, 1.8], [3.1, 3.9], [4.8, 6.1]])
        >>> dipolarity_param(field_map, fitted_map)
        0.9594001028529425
        >>> dipolarity_param(np.array([field_map, field_map]),
                             np.array([fitted_map, fitted_map]))
        array([0.9594001 , 0.9594001 ])

    References
    ----------
    .. [1] Fu, Roger R., Eduardo A. Lima, Michael W. R. Volk, and Raisa Trubko.
    “High-Sensitivity Moment Magnetometry With the Quantum Diamond Microscope.”
    Geochemistry, Geophysics, Geosystems 21, no. 8 (2020): e2020GC009147. https://doi.org/10/ghfpqv.
    """

    if field_map.shape != fitted_map.shape:
        raise ValueError("The shapes of the input arrays do not match.")

    residual: NDArray64 = fitted_map - field_map
    return 1 - (rms(residual) / rms(field_map))


@guvectorize(
    [(float64[:, :], float64[:])], "(m,n)->()", target="parallel", nopython=True
)
def rms(field_map: NDArray64, result: NDArray64) -> None:
    """
    Calculate the root mean square (RMS) of a magnetic field map or a collection of maps.

    Args:
        field_map (NDArray64): Magnetic field map to calculate the RMS of. The input can be a 2D array representing a single map,
                           or a 3D array representing multiple maps.

    Returns:
        np.float64: RMS of the magnetic field map(s). If the input is a 3D array, the output is a 1D array containing the RMS
                    of each map.

    Examples:
        >>> import numpy as np
        >>> field_map_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> rms(field_map_2d)
        3.8944404818493075

        >>> field_map_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        >>> rms(field_map_3d)
        array([2.73861279, 6.59545298])
    """

    result[0] = np.sqrt(np.mean(np.square(field_map)))


def upward_continue(
    field_map: NDArray64, distance: float, pixel_size: float, oversample: int = 2
) -> NDArray64:
    """Upward continues a map.

    Calculates a new map that is the upward continuation of the initial map by a given distance.
    In other words, it returns a new map that looks as if it was measured at a different distance from the sample.

    Args:
        field_map: The 2D array representing the map to be continued.
        distance: The distance to upward continue the map in m.
        pixel_size: The size of the pixel in the map in m.
        oversample: The oversampling factor to use. Default value is 2.

    Returns:
        The upward continued map as a 2D numpy array.

    Examples:
        >>> import numpy as np
        >>> field_map = np.array([[1, 2], [3, 4]], dtype=np.float64)
        >>> upward_continue(field_map, 2, 0.5)
        array([[ 3.77615757,  3.38899885],
               [ 4.74852508,  4.26136635]])
    """
    ypix, xpix = field_map.shape

    # pad the magnetic field map with zeros
    field_map = pad_map(field_map, oversample)
    new_x, new_y = field_map.shape

    # calculate the frequency coordinates
    x_steps = np.fft.fftfreq(new_x, pixel_size)
    y_steps = np.fft.fftfreq(new_y, pixel_size)
    fgrid_x, fgrid_y = np.meshgrid(x_steps, y_steps, indexing='ij')
    kx = 2 * np.pi * fgrid_x
    ky = 2 * np.pi * fgrid_y
    k = np.sqrt(kx**2 + ky**2)

    # Calculate the filter frequency response associated with the x component
    filter_response = np.exp(-distance * k)

    # Compute the FFT of the magnetic field map
    fft_map = np.fft.fft2(field_map)

    # Calculate the filtered map
    fft_filtered_map = fft_map * filter_response

    # Compute the inverse FFT of the filtered map
    filtered_map = np.fft.ifft2(fft_filtered_map).real

    # Crop the map to remove zero padding
    xcrop_start = (oversample-1) * ypix
    xcrop_end = xcrop_start + xpix
    ycrop_start = (oversample-1) * ypix
    ycrop_end = ycrop_start + xpix

    # Crop matrices to get rid of zero padding

    return filtered_map[ycrop_start:ycrop_end, xcrop_start:xcrop_end]



def pad_map(field_map: NDArray64, oversample: int = 2) -> NDArray64:
    """
    Pads a magnetic field map with zeros.

    Args:
        field_map (NDArray64): The magnetic field map to be padded.
        oversample (int, optional): The oversampling factor. The padding size will be
            (oversample - 1) times the original map dimensions. Defaults to 2.

    Returns:
        NDArray64: The padded magnetic field map.

    Examples:
        >>> import numpy as np
        >>> field_map = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        >>> pad_map(field_map)
            array([[0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 2., 0., 0.],
                   [0., 0., 3., 4., 0., 0.],
                   [0., 0., 5., 6., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.]])
    """
    new_shape = np.array(field_map.shape)
    new_shape[-2:] *= 1+ 2*(oversample - 1)
    padded = np.zeros(new_shape, dtype=field_map.dtype)
    center_row = (new_shape[-2] - field_map.shape[-2]) // 2
    center_col = (new_shape[-1] - field_map.shape[-1]) // 2

    padded[
        ...,
        center_row : center_row + field_map.shape[-2],
        center_col : center_col + field_map.shape[-1],
    ] = field_map
    return padded
