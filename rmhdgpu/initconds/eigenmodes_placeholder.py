"""Placeholder mode initializers used mainly for FFT indexing tests."""

from __future__ import annotations

import numpy as np
from typing import Any


def single_mode_field(
    grid: Any,
    backend: Any,
    mode_indices: tuple[int, int, int],
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Any:
    """Return a Fourier field with one stored `rfftn` mode populated.

    Parameters
    ----------
    mode_indices:
        Physical integer mode numbers `(n_x, n_y, n_z)`. The `z` mode must lie
        in the stored nonnegative range `0 <= n_z <= Nz//2`.

    Notes
    -----
    This is a placeholder testing utility for FFT indexing, not a physical
    eigenmode initializer.
    """

    nx, ny, nz = mode_indices
    if nz < 0 or nz > (grid.Nz // 2):
        raise ValueError(
            f"n_z must satisfy 0 <= n_z <= Nz//2; got n_z={nz} for Nz={grid.Nz}."
        )

    field_hat = backend.zeros(grid.fourier_shape, dtype=grid.complex_dtype)
    coeff = amplitude * np.exp(1j * phase)

    ix = nx % grid.Nx
    iy = ny % grid.Ny
    field_hat[ix, iy, nz] = coeff
    return field_hat

