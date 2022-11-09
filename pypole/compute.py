import numba
import numpy as np
from numpy.typing import NDArray


def dipolarity_param(data_map, fitted_map):
    """
    Calculate the dipolarity parameter of a magnetic dipole field.

    The dipolarity parameter (DP) is defined as the ratio of the rms of the
    residual (fitted_map - map) to the rms of the data map.
    DP was first introduced by [1]_.

    Parameters
    ----------
    data_map: ndarray(pixel, pixel)
        magnetic field map
    fitted_map: ndarray(pixel, pixel)
        fitted magnetic field map

    Returns
    -------
    dipolarity parameter: float
        dipolarity parameter of the magnetic dipole field

    References
    ----------
    .. [1] Fu, Roger R., Eduardo A. Lima, Michael W. R. Volk, and Raisa Trubko.
    “High-Sensitivity Moment Magnetometry With the Quantum Diamond Microscope.”
    Geochemistry, Geophysics, Geosystems 21, no. 8 (2020): e2020GC009147. https://doi.org/10/ghfpqv.

    """
    residual: NDArray = fitted_map - data_map
    return 1 - (rms(residual) / rms(data_map))


def rms(b_map: np.ndarray):
    """
    Calculate the root mean square of a map.

    """
    return np.sqrt(np.mean(np.square(b_map)))
