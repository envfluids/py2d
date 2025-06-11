import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from py2d.filter import coarse_spectral_filter_square_2DFHIT


def plot_omega_high_low(
    omega_high_res_hat: np.ndarray,
    omega_low_res_hat: np.ndarray,
    NX_low_res: int,
) -> tuple[mpl.figure.Figure, tuple[mpl.axes._axes.Axes]]:
    """Visualize the 2D Fourier transforms of the high and low resolution
    vorticities.

    Parameters
    ----------
    omega_high_res, omega_low_res
        2D Fourier transforms of vorticities resulting from simulations using
        `Py2D_solver`
    NX_low_res
        Spatial resolution of low resolution simulation

    Returns
    -------
    fig
        The figure
    ax_hi, cax, ax_lo
        The subplot and colorbar axes
    """
    # Filter the high resolution data to visualize what the difference, used in
    # nudging and parameter updating, looks like.
    omega_high_res_coarse_hat = coarse_spectral_filter_square_2DFHIT(
        omega_high_res_hat, NX_low_res
    )

    # Shift the data so that the zero wavenumber is in a corner instead of the
    # center.
    high = np.fft.fftshift(omega_high_res_coarse_hat.real)
    low = np.fft.fftshift(omega_low_res_hat.real)

    _, axs = fig, (ax_hi, cax, ax_lo) = plt.subplots(
        1, 3, figsize=(10, 4.25), width_ratios=(10, 1, 10)
    )
    cmap = mpl.cm.viridis
    vmin, vmax = np.min((high, low)), np.max((high, low))
    norm = mpl.colors.SymLogNorm(linthresh=10, vmin=vmin, vmax=vmax)
    ax_hi.imshow(high, cmap=cmap, norm=norm)
    ax_lo.imshow(low, cmap=cmap, norm=norm)

    map = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(map, cax=cax)
    fig.tight_layout()
    return fig, axs
