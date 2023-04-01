import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray

from pypole import NDArray64
from typing import Tuple

"""
Converts between xyz and polar/azimuth coordinates with the convention of N along -Y-axis and +Z down.

                  N, -Y
               *********
           *                  *
        *                         *
      *                             *
   -X *     ↑ -Z ↓           +Z       *  +X (declination = 90°)
      *                             *
       *                         *
          *                  *
             *********
               S, +Y (declination = 180°)

Each point is represented by a vector (x, y, z), where x, y, and z are the coordinates
in a right-handed Cartesian coordinate system with N pointing in the -y direction,
+Z pointing down, and +X pointing to the right. Polar/azimuth coordinates are represented
as (declination, inclination, magnitude), where declination is the angle in degrees in
the x-y plane measured from the x-axis to the projection of the vector on the x-y plane,
inclination is the angle in degrees between the vector and the z-axis, and magnitude is
the magnitude of the vector.

Functions:
----------
xyz2dim(xyz: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    Converts xyz coordinates to polar/azimuth coordinates.
dim2xyz(dim: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    Converts polar/azimuth coordinates to xyz coordinates.


Examples:
----------
>>> import numpy as np
>>> from pypole import convert
>>> xyz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
>>> convert.xyz2dim(xyz)
array([[  0., -90.,   1.],
       [ 90., -90.,   1.],
       [  0.,   0.,   1.]])

>>> dim = np.array([[0, 0, 1], [0, 90, 1]])
>>> convert.dim2xyz(dim)
array([[ 0., -1.,  0.],
       [ 0.,  0., -1.]])

Note: In the polar/azimuth convention used by this module, the declination is measured counterclockwise
from the x-axis, with 0 degrees pointing towards the +y direction and 90 degrees pointing towards the
+x direction. The inclination is measured from the z-axis, with positive angles pointing downwards.
This convention is illustrated by the circle diagram above.
"""

from numba import guvectorize


@guvectorize(["void(float64[:], float64[:])"], "(n)->(n)")
def dim2xyz(dim: Tuple[float, float, float], xyz: Tuple[float, float, float]):
    """
    A guvectorize function that calculates x,y,z from polar/azimuth data.

    Parameters
    ----------
    dim: tuple
        A tuple containing declination, inclination, magnitude.
    xyz: tuple
        A tuple to hold the calculated x,y,z values.

    """
    # separate columns and convert to radians
    dec = np.deg2rad(dim[0])
    inc = np.deg2rad(dim[1])
    mag = dim[2]

    xyz[0] = mag * np.sin(dec) * np.cos(inc)
    xyz[1] = -mag * np.cos(dec) * np.cos(inc)
    xyz[2] = mag * np.sin(-inc)


@guvectorize(["void(float64[:], float64[:])"], "(n)->(n)")
def xyz2dim(xyz: Tuple[float, float, float], dim: Tuple[float, float, float]):
    """
    Calculates polar/azimuth from x,y,z data.

    Parameters
    ----------
    xyz: tuple of floats (3,)
        x,y,z values of a vector [x,y,z]
    dim: tuple of floats (3,)
        declination, inclination, magnitude

    Notes
    -----
    The declination and inclination values are returned in degrees.

    The input `xyz` must be an array-like object with shape (n, 3), where n is the number of vectors.

    Returns
    -------
    ndarray
        An array of shape (n, 3) with the polar/azimuthal coordinates for each input vector.

    Examples
    --------
    >>> xyz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> xyz2dim(xyz)
    array([[  0.,  90.,   1.],
           [ 90.,   0.,   1.],
           [  0.,  -90.,   1.]])
    """
    x, y, z = xyz

    # calculate dec and map to 0-360 degree range
    dim[0] = (90 + np.degrees(np.arctan2(y, x))) % 360
    dim[2] = np.linalg.norm(xyz)
    dim[1] = -np.degrees(np.arcsin(z / dim[2]))


def main():
    dim = np.array([[0, 0, 1], [0, 10, 1], [90, 45, 2]])
    xyz = dim2xyz(dim)
    print(xyz2dim(xyz))


if __name__ == "__main__":
    main()
