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
from mpl_toolkits.axes_grid1 import ImageGrid

from pypole import NDArray64, compute, fit, maps


def plot_map(
    map_data: NDArray64,
    title: str = "",
    cbar_label: str = "B [T]",
    vminmax=None,
    ax=None,
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

    cax = ax.imshow(map_data, origin="lower")
    ax.set(title=title, xlabel="px", ylabel="px")
    cbar = plt.colorbar(cax, ax=ax, label=cbar_label)
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

    fig = plt.figure(figsize=(10, 3))
    fig.suptitle(title)

    res = map1 - map2

    ax = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 3),
        axes_pad=0.05,
        cbar_location="right",
        cbar_mode="edge",
        cbar_size="5%",
        cbar_pad=0.05,
    )
    # set vmin and vmax to 0.01% and 99.99% percentile of the data map
    vmin, vmax = np.percentile(map1, [0.01, 99.99])

    plot_map(map1, title="data map", cbar_label="B [T]", vminmax=(vmin, vmax), ax=ax[0])
    plot_map(
        map2, title="fitted map", cbar_label="B [T]", vminmax=(vmin, vmax), ax=ax[1]
    )
    plot_map(res, title="residual", cbar_label="B [T]", vminmax=(vmin, vmax), ax=ax[2])

    print(f"residual: max: {np.max(res):.2e} std: {np.std(res):.2e} T")
    print(f"dipolarity parameter: {compute.dipolarity_param(map1, map2):.2f}")
