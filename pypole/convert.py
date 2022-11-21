import numba
import numpy as np
from numpy.typing import NDArray

"""
Converts between xyz and polar/azimuth coordinates with the convention of N along y-axis and z along the down direction.
"""


def dim2xyz(dim: NDArray) -> NDArray[np.float64]:
    """calculates x,y,z from polar/azimuth data

    Parameters
    ----------
    dim: ndarray
        declination, inclination, magnitude

    Returns
    -------
    np.array
        of x,y,z values

    >>> dim2xyz(np.array([[0,0,1], [0,90,1]]))
    array([[ 0., -1.,  0.], [ 0.,  0., -1.]])
    """
    # convert to np array
    dim = np.atleast_2d(dim)

    # separate columns
    # convert to radians
    dec = np.deg2rad(dim[:, 0])
    inc = np.deg2rad(dim[:, 1])
    mag = dim[:, 2]

    moment_vector = [
        mag * np.sin(dec) * np.cos(inc),
        -mag * np.cos(dec) * np.cos(inc),
        mag * np.sin(-inc),
    ]
    return np.array(moment_vector).T


def xyz2dim(xyz: NDArray) -> NDArray[np.float64]:
    """calculates polar/azimuth from x,y,z data

    Parameters
    ----------
    xyz: ndarray (3,n)
        x,y,z values of n vectors [[x1,y1,z1],[x2,y2,z2],...]
    """
    xyz = np.atleast_2d(xyz)

    # calculate dec and map to 0-360 degree range
    dec = xy2dec(xyz[:, 0:2])
    mag = np.linalg.norm(xyz, axis=1)
    inc = z2inc(z=xyz[:, 2], mag=mag)

    return np.stack([dec, inc, mag]).T


def xy2dec(xy) -> NDArray[np.float64]:
    """calculates declination from x,y data
    Parameters
    ----------
    xy: ndarray (2,n)
        x,y values of n vectors [[x1,y1],[x2,y2],...]
    """
    xy = np.atleast_2d(xy)
    x = xy[:, 0]
    y = xy[:, 1]
    return (90 + np.degrees(np.arctan2(y, x))) % 360


def z2inc(z, mag) -> NDArray[np.float64]:
    """calculates inclination from z data

    Parameters
    ----------
    z: ndarray (n,)
        z values of n vectors [z1,z2,...]
    mag: ndarray (n,)
        magnitude of n vectors [m1,m2,...]
    """
    return -np.degrees(np.arcsin(z / mag))


def main():
    dim = np.array([[0, 0, 1], [0, 10, 1], [90, 45, 2]])
    xyz = dim2xyz(dim)
    print(xyz2dim(xyz))


if __name__ == "__main__":
    main()
