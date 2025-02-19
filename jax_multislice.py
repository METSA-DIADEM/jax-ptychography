import jax
import jax.numpy as jnp
from ase import units
from functools import partial

def move_probe(probe, new_pos):
    """
    Move the probe by a given shift.

    Args:
        probe: The probe to move.
        shift: The shift to apply to the probe.

    Returns:
        The probe after the shift.
    """
    current_pos_row = probe.shape[0]//2 #probe coordinates are in y, x
    current_pos_col = probe.shape[1]//2 #probe coordinates are in y, x
    new_pos_row = new_pos[0] #scan coordinates are given in x, y
    new_pos_col = new_pos[1] #scan coordinates are given in x, y

    # if probe.shape[0] % 2 == 1:
    #     current_pos_row += 1

    shift_to_row = new_pos_row - current_pos_row
    shift_to_col = new_pos_col - current_pos_col
    shift = jnp.array([shift_to_row, shift_to_col])

    return jnp.roll(probe, shift, axis=(0, 1))


def propagation_kernel(n: int,
                       m: int,
                       ps: float,
                       z: float,
                       energy: float):
    wavelength = energy2wavelength(energy)

    fx = jnp.fft.fftfreq(n, ps[0])
    fy = jnp.fft.fftfreq(m, ps[1])
    Fx, Fy = jnp.meshgrid(fx, fy, indexing='ij')

    H = jnp.exp(1j * (
        2 * jnp.pi / wavelength) * z) * jnp.exp(
        -1j * jnp.pi * wavelength * z * (Fx**2 + Fy**2))

    return H


@jax.jit
def FresnelPropagator(u, H):
    ufft = jnp.fft.fft2(u)
    u_prop = jnp.fft.ifft2(H * ufft)
    return u_prop


@jax.jit
def transmission_function(array, energy):
    sigma = energy2sigma(energy)
    return jnp.exp(1j * sigma * array)


@jax.jit
def relativistic_mass_correction(energy: float) -> float:
    return 1 + units._e * energy / (units._me * units._c**2)


@jax.jit
def energy2mass(energy: float) -> float:
    """
    Calculate relativistic mass from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic mass [kg]
    """
    return relativistic_mass_correction(energy) * units._me


@jax.jit
def energy2sigma(energy: float) -> float:
    """
    Calculate interaction parameter from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Interaction parameter [1 / (Å * eV)].
    """
    return (
        2 * jnp.pi * energy2mass(energy) * units.kg * units._e * units.C *
        energy2wavelength(energy) /
        (units._hplanck * units.s * units.J) ** 2
    )


@jax.jit
def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic de Broglie wavelength [Å].
    """
    return (
        units._hplanck
        * units._c
        / jnp.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    ).astype(jnp.float32)
