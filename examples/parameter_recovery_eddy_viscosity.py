from jax import numpy as jnp

from py2d.convert import Omega2Psi_2DFHIT_spectral, Psi2UV_2DFHIT_spectral
from py2d.eddy_viscosity_models import nabla_squared_omega
from py2d.filter import coarse_spectral_filter_square_2DFHIT
from py2d.initialize import initialize_wavenumbers_2DFHIT
from py2d.SGSModel import SGSModel


def calculate_difference(
    omega_high_res_hat: jnp.ndarray, omega_low_res_hat: jnp.ndarray, mu: float
) -> jnp.ndarray:
    """Calculate the difference between the high and low resolution simulations.

    The high resolution state is filtered before comparing with the low
    resolution state.

    Parameters
    ----------
    omega_high_res, omega_low_res
        2D Fourier transforms of vorticities resulting from simulations using
        `Py2D_solver`
    mu
        Nudging parameter, a multiplicative factor adjusting how much the low
        resolution simulation is pushed toward the high resolution one

        Must be positive.

    Returns
    -------
    difference
        The scaled difference between the Fourier transforms of the vorticities,
        mu * (high - low).
    """
    NX_low_res = omega_low_res_hat.shape[0]

    # Coarse graining the high resolution simulation grid to the low resolution
    # simulation grid by removing high-wavenumber data.
    # It coarse grains to Ngrid = NX_low_res.
    Omega_high_res_coarse_hat = coarse_spectral_filter_square_2DFHIT(
        omega_high_res_hat, NX_low_res
    )

    difference = mu * (Omega_high_res_coarse_hat - omega_low_res_hat)
    return difference


def compute_eddy_viscosity_coeff_step(
    sgs_model_type: str,
    omega_high_res: jnp.ndarray,
    omega_low_res: jnp.ndarray,
    NX_low_res: int,
    current_eddy_viscosity_coeff: float,
    mu: float,
    lr: float,
) -> float:
    """Compute the step to add to the eddy viscosity coefficient to optimize it.

    Parameters
    ----------
    sgs_model_type
        The type of SGS model to use; see `py2d/SGSModel.py`.
    omega_high_res, omega_low_res
        Vorticities resulting from simulations using `Py2D_solver`
    NX_low_res
        Spatial resolution of low resolution simulation
    current_eddy_viscosity_coeff
        Current eddy viscosity coefficient as part of eddy viscosity according
        to SGS model
    mu
        Nudging parameter, a multiplicative factor adjusting how much the low
        resolution simulation is pushed toward the high resolution one

        Must be positive.
    lr
        Learning rate, a multiplicative factor adjusting the rate of gradient
        descent

    Returns
    -------
    step
        The quantity to add to the eddy viscosity coefficient, e.g.,
        `new_eddy_viscosity_coeff += step`
    """
    omega_high_res_hat = jnp.fft.fft2(omega_high_res)
    omega_low_res_hat = jnp.fft.fft2(omega_low_res)

    Lx = 2 * jnp.pi  # Domain length
    Kx, Ky, _, Ksq, invKsq = initialize_wavenumbers_2DFHIT(
        NX_low_res, NX_low_res, Lx, Lx
    )

    PiOmega_eddy_viscosity_model = _create_sgs_model(
        sgs_model_type,
        NX_low_res,
        current_eddy_viscosity_coeff,
        Lx,
        Kx,
        Ky,
        Ksq,
    )

    _update_sgs_model(
        PiOmega_eddy_viscosity_model, omega_low_res_hat, Kx, Ky, invKsq
    )

    step = _compute_step(
        sgs_model_type,
        PiOmega_eddy_viscosity_model,
        omega_high_res_hat,
        omega_low_res_hat,
        Kx,
        Ky,
        mu,
        lr,
    )
    return step


def _create_sgs_model(
    sgs_model_type: str,
    NX_low_res: int,
    current_eddy_viscosity_coeff: float,
    Lx: float,
    Kx: jnp.ndarray,
    Ky: jnp.ndarray,
    Ksq: jnp.ndarray,
) -> SGSModel:
    Delta = 2 * Lx / NX_low_res  # Filter Width
    PiOmega_eddy_viscosity_model = SGSModel(
        Kx, Ky, Ksq, Delta, sgs_model_type, current_eddy_viscosity_coeff
    )
    return PiOmega_eddy_viscosity_model


def _update_sgs_model(
    PiOmega_eddy_viscosity_model: SGSModel,
    omega_low_res_hat: jnp.ndarray,
    Kx: jnp.ndarray,
    Ky: jnp.ndarray,
    invKsq: jnp.ndarray,
) -> None:
    """Update the SGS model *in-place* using its method `calculate`."""
    Psi_hat = Omega2Psi_2DFHIT_spectral(omega_low_res_hat, invKsq)
    U1_hat, V1_hat = Psi2UV_2DFHIT_spectral(Psi_hat, Kx, Ky)
    PiOmega_eddy_viscosity_model.update_state(
        Psi_hat, omega_low_res_hat, U1_hat, V1_hat
    )
    PiOmega_eddy_viscosity_model.calculate()


def _compute_step(
    sgs_model_type: str,
    PiOmega_eddy_viscosity_model: SGSModel,
    omega_high_res_hat: jnp.ndarray,
    omega_low_res_hat: jnp.ndarray,
    Kx: jnp.ndarray,
    Ky: jnp.ndarray,
    mu: float,
    lr: float,
) -> float:
    """Based on the section "Eddy Viscosity Models" in
    `parameter_recovery_eddy_viscosity.ipynb`.
    """
    eddy_viscosity = PiOmega_eddy_viscosity_model.eddy_viscosity
    eddy_viscosity_coeff = PiOmega_eddy_viscosity_model.C_MODEL

    # Compute the derivative of the equation model with respect to the parameter
    # to be optimized.
    laplacian = nabla_squared_omega(omega_low_res_hat, Kx, Ky)
    if sgs_model_type == "LEITH":
        model_wrt_parameter = (
            -eddy_viscosity * laplacian / eddy_viscosity_coeff**3
        )
    else:
        raise NotImplementedError()

    # Compute the asymptotic approximation of the sensitivity.
    asymptotic_approximation = -1 / mu * model_wrt_parameter

    # Compute the derivative of the error with respect to the parameter.
    difference = -calculate_difference(omega_high_res_hat, omega_low_res_hat, 1)
    error_wrt_parameter = jnp.sum(
        jnp.real(difference.conj() * asymptotic_approximation)
    )

    step = -lr * error_wrt_parameter
    return step
