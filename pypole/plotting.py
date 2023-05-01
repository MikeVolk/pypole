"""
plotting.py - Contains functions for plotting magnetic field maps and dipole fits.

Functions:
- plot_map(field_map: ndarray, title: str = "") -> None
- plot_dipoles_on_map(
        field_map: ndarray,
        dipole_params: Optional[Union[ndarray, list[ndarray]]] = None,
        title: str = "",
        alpha: float = 1,
        scale: Optional[float] = None,
    ) -> None
- plot_fit_results(
        field_map: ndarray,
        dipole_params: Union[ndarray, list[ndarray]],
        title: Optional[str] = None,
    ) -> None
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

from pypole import NDArray64, compute, fit, maps


def get_cax(ax):
    """
    Add a colorbar to an AxesImage object and position it next to an existing Axes object.

    Parameters
    ----------
    ax_image : AxesImage
        The AxesImage object that the colorbar will correspond to.
    ax : Axes
        The Axes object that the colorbar will be positioned next to.

    Returns
    -------
    cax : Axes
        The Axes object for the colorbar.

    """
    # Use make_axes_locatable to create a Divider object for positioning the colorbar
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="5%", pad=0.05)


def plot_map(
    map_data: NDArray64,
    title: str = "",
    cbar_label: str = "B [T]",
    vminmax=None,
    ax=None,
    show_cbar=True,
) -> None:
    """Plot a single map

    Args:
        map_data (NDArray64): The map data to plot
        title (str, optional): Title of the plot. Defaults to None.
        cbar_label (str, optional): Label of the colorbar. Defaults to 'B [T]'.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    # set vmin and vmax to 0.01% and 99.99% percentile of the data map
    if vminmax is None:
        vmin, vmax = np.percentile(map_data, [0.01, 99.99])

    img = ax.imshow(map_data, origin="lower")
    ax.set(title=title, xlabel="px", ylabel="px")

    if show_cbar:
        cax = get_cax(ax)
        cbar = plt.colorbar(img, cax=cax, label=cbar_label)
        cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1e"))
    return ax


def compare_maps(map1: NDArray64, map2: NDArray64, title: str = "") -> None:
    """Plot a comparison of the data map and the fitted map

    Args:
        map1 (NDArray64): Map 1 to compare
        map2 (NDArray64): Map 2 to compare
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        None
    """

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle(title)

    res = map1 - map2

    # set vmin and vmax to 0.01% and 99.99% percentile of the data map
    vmin, vmax = np.percentile(map1, [0.01, 99.99])

    plot_map(
        map1, title="data map", cbar_label="B [T]", vminmax=(vmin, vmax), ax=ax[0], show_cbar=False
    )
    plot_map(
        map2,
        title="fitted map",
        cbar_label="B [T]",
        vminmax=(vmin, vmax),
        ax=ax[1],
        show_cbar=False,
    )
    plot_map(res, title="residual", cbar_label="B [T]", vminmax=(vmin, vmax), ax=ax[2])

    print(f"residual: max: {np.max(res):.2e} std: {np.std(res):.2e} T")
    print(f"dipolarity parameter: {compute.dipolarity_param(map1, map2):.2f}")
