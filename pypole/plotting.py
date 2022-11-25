import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from pypole import NDArray64, compute, fit, maps


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
    vmin, vmax = np.percentile(map1, [0, 100])
    ax[0].imshow(map1, vmin=vmin, vmax=vmax, origin="lower")
    ax[0].set(title="data map", xlabel="px", ylabel="px")
    ax[1].imshow(map2, vmin=vmin, vmax=vmax, origin="lower")
    ax[1].set(title="fitted map", xlabel="px", ylabel="px")
    l = ax[2].imshow(res, vmin=vmin, vmax=vmax, origin="lower")
    ax[2].set(title="residual", xlabel="px", ylabel="px")

    plt.colorbar(l, cax=ax.cbar_axes[0], label="B [T]")
    plt.tight_layout()
    plt.show()

    print(f"residual: max: {np.max(res):.2e} std: {np.std(res):.2e} T")
    print(f"dipolarity parameter: {compute.dipolarity_param(map1, map2):.2f}")
