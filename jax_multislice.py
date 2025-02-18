import jax
import jax.numpy as jnp
from ase import units
from functools import partial

@partial(jax.jit, static_argnums=(1, 2, 3))
def forward(slices,
            probe,
            propagation_kernel,
            scan_position_px):

    """
    Forward model for a multislice simulation.

    Args:
        slices: The 3D array representing the object potential.
        probe: The probe to use for the simulation.
        propagation_kernel: The kernel used for propagation between slices.
        scan_position_px: The scan position in pixels.

    Returns:
        The simulated diffraction pattern.
    """
    probe_shifted = jnp.roll(probe, scan_position_px, axis=(0, 1))
    exit_wave = probe_shifted * slices[0]

    def body_fun(i, exit_wave):
        exit_wave = FresnelPropagator(exit_wave, propagation_kernel)
        exit_wave = exit_wave * slices[i]
        return exit_wave

    exit_wave = jax.lax.fori_loop(1,
                                  slices.shape[0],
                                  body_fun,
                                  exit_wave)

    diffraction_pattern = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(exit_wave)))
    return diffraction_pattern

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def forward_with_scan_positions(slices,
                                probe,
                                scan_positions_px,
                                energy,
                                sampling,
                                slice_thickness):
    """
    Forward model for a multislice simulation with scan positions.

    Args:
        slices: The 3D array representing the object potential.
        probe: The probe to use for the simulation.
        scan_positions_px: The scan positions in pixels.
        energy: The energy of the incident wave in eV.
        sampling: The sampling intervals in the x and y directions.
        slice_thickness: The thickness of each slice.

    Returns:
        The simulated diffraction patterns for the given scan positions.
    """
    n, m = slices.shape[1::]
    scan_positions_shape = scan_positions_px.shape[0:2]
    scan_positions_px = scan_positions_px.reshape(-1, 2)
    H = propagation_kernel(n, m, sampling, slice_thickness, energy)
    transmission = transmission_function(slices, energy)

    diffraction_patterns = jax.vmap(lambda pos: forward(transmission, probe, H, pos))(scan_positions_px)
    diffraction_patterns = diffraction_patterns.reshape((*scan_positions_shape, n, m))
    return diffraction_patterns


def move_probe(probe, shift):
    """
    Move the probe by a given shift.

    Args:
        probe: The probe to move.
        shift: The shift to apply to the probe.

    Returns:
        The probe after the shift.
    """
    new_row = -(shift[1] - probe.shape[1]//2)
    new_col = shift[0] - probe.shape[0]//2
    shift = jnp.array([new_row, new_col])

    return jnp.roll(probe, shift, axis=(0, 1))


def propagation_kernel(n: int,
                       m: int,
                       ps: float,
                       z: float,
                       energy: float):
    wavelength = energy2wavelength(energy)

    fx = jnp.fft.fftfreq(n, ps[0])
    fy = jnp.fft.fftfreq(m, ps[1])
    Fx, Fy = jnp.meshgrid(fy, fx)

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
