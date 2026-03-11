"""Spectral masking utilities."""

from __future__ import annotations

from typing import Any


def build_dealias_mask(grid: Any, backend: Any, mode: str = "two_thirds") -> Any:
    """Build a boolean mask for dealiasing in `rfftn` storage.

    For `mode="two_thirds"`, the retained integer Fourier indices satisfy

    - `|n_x| < Nx / 3`
    - `|n_y| < Ny / 3`
    - `n_z < Nz / 3`

    where `n_z` refers to the stored nonnegative half-spectrum of the real FFT.
    """

    if mode != "two_thirds":
        raise ValueError(f"Unsupported dealias mode {mode!r}.")

    xp = backend.xp

    nx = xp.fft.fftfreq(grid.Nx) * grid.Nx
    ny = xp.fft.fftfreq(grid.Ny) * grid.Ny
    nz = xp.arange(grid.Nz // 2 + 1)

    keep_x = xp.abs(nx) < (grid.Nx / 3.0)
    keep_y = xp.abs(ny) < (grid.Ny / 3.0)
    keep_z = nz < (grid.Nz / 3.0)

    mask = (
        keep_x.reshape(grid.Nx, 1, 1)
        & keep_y.reshape(1, grid.Ny, 1)
        & keep_z.reshape(1, 1, grid.Nz // 2 + 1)
    )
    return mask


def apply_mask(f_hat: Any, mask: Any) -> Any:
    """Apply a boolean mask in place and return the same array."""

    f_hat *= mask
    return f_hat

