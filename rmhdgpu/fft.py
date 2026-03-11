"""Thin wrappers around `rfftn` / `irfftn`."""

from __future__ import annotations

from typing import Any


class FFTManager:
    """Wrap backend FFT calls with one consistent convention.

    Backend behavior:

    - `numpy`: NumPy arrays with `numpy.fft`
    - `scipy_cpu`: NumPy arrays with `scipy.fft`, optionally using
      `backend.fft_workers`
    - `cupy`: CuPy arrays with `cupy.fft`

    All backends use the same default FFT normalization:

    - forward transform: unnormalized
    - inverse transform: scaled by `1 / (Nx * Ny * Nz)`

    With that convention, `c2r(r2c(f))` recovers the original real-space field
    to numerical precision when the Fourier array is compatible with `rfftn`
    storage.
    """

    def __init__(self, grid: Any, backend: Any) -> None:
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp

    def r2c(self, f_real: Any, out: Any | None = None) -> Any:
        """Transform a real field to `rfftn` Fourier storage."""

        if self.backend.backend_name == "scipy_cpu":
            transformed = self.backend.scipy_fft.rfftn(
                f_real,
                s=self.grid.real_shape,
                axes=(0, 1, 2),
                workers=self.backend.fft_workers,
            )
        else:
            transformed = self.xp.fft.rfftn(
                f_real,
                s=self.grid.real_shape,
                axes=(0, 1, 2),
            )
        if out is not None:
            out[...] = transformed
            return out
        return transformed

    def c2r(self, f_hat: Any, out: Any | None = None) -> Any:
        """Transform an `rfftn` Fourier field back to real space."""

        if self.backend.backend_name == "scipy_cpu":
            transformed = self.backend.scipy_fft.irfftn(
                f_hat,
                s=self.grid.real_shape,
                axes=(0, 1, 2),
                workers=self.backend.fft_workers,
            )
        else:
            transformed = self.xp.fft.irfftn(
                f_hat,
                s=self.grid.real_shape,
                axes=(0, 1, 2),
            )
        if out is not None:
            out[...] = transformed
            return out
        return transformed
