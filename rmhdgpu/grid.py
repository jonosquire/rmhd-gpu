"""Periodic grid and Fourier-space wavenumber setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Grid:
    """Geometry and spectral metadata for a fully periodic 3D box.

    Notes
    -----
    - The guide-field / parallel direction is always the last axis, `z`.
    - Fourier arrays use `rfftn` storage with shape `(Nx, Ny, Nz//2 + 1)`.
    - `inv_kperp2` is regularized by setting the `k_perp = 0` modes to zero.
      This keeps inverse perpendicular Laplacian operations finite everywhere.
    """

    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float
    dx: float
    dy: float
    dz: float
    real_shape: tuple[int, int, int]
    fourier_shape: tuple[int, int, int]
    real_dtype: np.dtype
    complex_dtype: np.dtype
    x: Any
    y: Any
    z: Any
    kx: Any
    ky: Any
    kz: Any
    kperp2: Any
    kpar2: Any
    k2: Any
    inv_kperp2: Any
    mask_kperp0: Any


def build_grid(config: Any, backend: Any) -> Grid:
    """Construct a real-space grid and its matching `rfftn` wavenumber arrays."""

    xp = backend.xp

    dx = config.Lx / config.Nx
    dy = config.Ly / config.Ny
    dz = config.Lz / config.Nz

    real_shape = (config.Nx, config.Ny, config.Nz)
    fourier_shape = (config.Nx, config.Ny, config.Nz // 2 + 1)

    x = xp.arange(config.Nx, dtype=config.real_dtype) * dx
    y = xp.arange(config.Ny, dtype=config.real_dtype) * dy
    z = xp.arange(config.Nz, dtype=config.real_dtype) * dz

    kx_1d = 2.0 * np.pi * xp.fft.fftfreq(config.Nx, d=dx)
    ky_1d = 2.0 * np.pi * xp.fft.fftfreq(config.Ny, d=dy)
    kz_1d = 2.0 * np.pi * xp.fft.rfftfreq(config.Nz, d=dz)

    kx = kx_1d.reshape(config.Nx, 1, 1).astype(config.real_dtype, copy=False)
    ky = ky_1d.reshape(1, config.Ny, 1).astype(config.real_dtype, copy=False)
    kz = kz_1d.reshape(1, 1, config.Nz // 2 + 1).astype(config.real_dtype, copy=False)

    kperp2_base = kx * kx + ky * ky
    kperp2 = xp.broadcast_to(kperp2_base, fourier_shape).copy()
    kpar2 = xp.broadcast_to(kz * kz, fourier_shape).copy()
    k2 = kperp2 + kpar2

    mask_kperp0 = kperp2 == 0.0
    inv_kperp2 = xp.zeros(fourier_shape, dtype=config.real_dtype)
    nonzero = ~mask_kperp0
    inv_kperp2[nonzero] = 1.0 / kperp2[nonzero]

    return Grid(
        Nx=config.Nx,
        Ny=config.Ny,
        Nz=config.Nz,
        Lx=config.Lx,
        Ly=config.Ly,
        Lz=config.Lz,
        dx=dx,
        dy=dy,
        dz=dz,
        real_shape=real_shape,
        fourier_shape=fourier_shape,
        real_dtype=config.real_dtype,
        complex_dtype=config.complex_dtype,
        x=x,
        y=y,
        z=z,
        kx=kx,
        ky=ky,
        kz=kz,
        kperp2=kperp2,
        kpar2=kpar2,
        k2=k2,
        inv_kperp2=inv_kperp2,
        mask_kperp0=mask_kperp0,
    )

