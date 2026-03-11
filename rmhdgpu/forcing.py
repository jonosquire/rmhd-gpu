"""Stochastic white-in-time forcing helpers.

The forcing implemented here is additive and refreshed every timestep. For one
field `q_i`, over a step `dt`, the kick is

`delta q_i = sigma_i * sqrt(dt) * xi_i`

where `sigma_i` is the configured RMS forcing strength in field-units per
`sqrt(time)`, and `xi_i` is a freshly generated band-limited, unit-RMS real
Gaussian field.

The forcing band is defined in integer Fourier mode-number magnitude

`n = sqrt(nx^2 + ny^2 + nz^2)`

rather than physical `k`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rmhdgpu.state import State


def mode_number_magnitude(grid: Any, backend: Any) -> Any:
    """Return the integer mode-number magnitude on the stored `rfftn` grid."""

    xp = backend.xp
    nx = xp.fft.fftfreq(grid.Nx) * grid.Nx
    ny = xp.fft.fftfreq(grid.Ny) * grid.Ny
    nz = xp.fft.rfftfreq(grid.Nz) * grid.Nz
    return xp.sqrt(
        nx.reshape(grid.Nx, 1, 1) ** 2
        + ny.reshape(1, grid.Ny, 1) ** 2
        + nz.reshape(1, 1, grid.Nz // 2 + 1) ** 2
    )


def forcing_shell_mask(
    grid: Any,
    backend: Any,
    n_min_force: float,
    n_max_force: float,
) -> Any:
    """Return the forcing-band mask on the Fourier grid."""

    n_mag = mode_number_magnitude(grid, backend)
    return (n_mag >= float(n_min_force)) & (n_mag <= float(n_max_force))


def shaped_random_real_field(
    grid: Any,
    backend: Any,
    fft: Any,
    *,
    n_min_force: float,
    n_max_force: float,
    alpha_force: float,
    rng: np.random.Generator,
) -> tuple[Any, Any]:
    """Return a unit-RMS real field and its Fourier transform.

    Construction:

    1. draw a Gaussian random real field in real space
    2. transform to `rfftn` storage
    3. apply the forcing shell mask
    4. apply amplitude shaping proportional to `n^{-alpha_force}`
    5. transform back to real space and normalize to unit RMS
    6. transform back to Fourier space

    The final Fourier field is masked again to remove roundoff-level leakage
    outside the selected forcing band.
    """

    real_noise_np = rng.standard_normal(grid.real_shape).astype(grid.real_dtype, copy=False)
    real_noise = backend.asarray(real_noise_np, dtype=grid.real_dtype)

    n_mag = mode_number_magnitude(grid, backend)
    band_mask = forcing_shell_mask(grid, backend, n_min_force, n_max_force)
    n_safe = backend.xp.where(n_mag > 0.0, n_mag, 1.0)
    shaping = backend.xp.where(band_mask, n_safe ** (-float(alpha_force)), 0.0)

    noise_hat = fft.r2c(real_noise)
    shaped_hat = noise_hat * shaping
    shaped_real = fft.c2r(shaped_hat)

    rms = backend.scalar_to_float(backend.xp.sqrt(backend.xp.mean(shaped_real**2)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise RuntimeError(
            "Stochastic forcing produced a zero or non-finite filtered field. "
            "Check the forcing band and spectral shaping parameters."
        )

    shaped_real = shaped_real / rms
    shaped_hat = fft.r2c(shaped_real)
    shaped_hat[...] *= band_mask
    return shaped_real, shaped_hat


def generate_forcing_kick(
    state: State,
    grid: Any,
    fft: Any,
    backend: Any,
    config: Any,
    rng: np.random.Generator,
    dt: float,
) -> State:
    """Return the additive stochastic forcing increment for one timestep.

    Each field is forced independently with a fresh random realization. The
    increment scales as `sqrt(dt)` so the configured amplitudes are interpreted
    as RMS forcing strengths in field-units per `sqrt(time)`.
    """

    kick = state.zeros_like()
    if not getattr(config, "use_forcing", False):
        return kick

    scale_dt = float(np.sqrt(dt))
    for field_name in kick.field_names:
        sigma = float(getattr(config, "force_amplitudes")[field_name])
        if sigma == 0.0:
            continue

        _, xi_hat = shaped_random_real_field(
            grid,
            backend,
            fft,
            n_min_force=float(getattr(config, "n_min_force")),
            n_max_force=float(getattr(config, "n_max_force")),
            alpha_force=float(getattr(config, "alpha_force")),
            rng=rng,
        )
        kick[field_name][...] = sigma * scale_dt * xi_hat

    return kick


def apply_forcing_kick(state: State, forcing_kick: State) -> State:
    """Return `state + forcing_kick` without mutating the input state."""

    result = state.copy()
    for field_name in result.field_names:
        result[field_name][...] += forcing_kick[field_name]
    return result
