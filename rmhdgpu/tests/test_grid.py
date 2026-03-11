from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.grid import build_grid


def test_grid_shapes_and_broadcast_arrays() -> None:
    config = Config(Nx=8, Ny=10, Nz=12, Lx=2.0 * np.pi, Ly=4.0 * np.pi, Lz=6.0 * np.pi)
    backend = build_backend(config)
    grid = build_grid(config, backend)

    assert grid.real_shape == (8, 10, 12)
    assert grid.fourier_shape == (8, 10, 7)
    assert grid.kx.shape == (8, 1, 1)
    assert grid.ky.shape == (1, 10, 1)
    assert grid.kz.shape == (1, 1, 7)
    assert grid.kperp2.shape == grid.fourier_shape
    assert grid.k2.shape == grid.fourier_shape


def test_inv_kperp2_is_finite_and_zero_on_kperp0_modes() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    inv_kperp2 = backend.to_numpy(grid.inv_kperp2)
    mask_kperp0 = backend.to_numpy(grid.mask_kperp0)

    assert np.isfinite(inv_kperp2).all(), "inv_kperp2 should be finite everywhere."
    assert np.all(inv_kperp2[mask_kperp0] == 0.0), "k_perp = 0 modes must be regularized to zero."


def test_small_grid_frequencies_match_fft_convention() -> None:
    config = Config(Nx=8, Ny=8, Nz=8, Lx=2.0 * np.pi, Ly=2.0 * np.pi, Lz=2.0 * np.pi)
    backend = build_backend(config)
    grid = build_grid(config, backend)

    kx = backend.to_numpy(grid.kx[:, 0, 0])
    ky = backend.to_numpy(grid.ky[0, :, 0])
    kz = backend.to_numpy(grid.kz[0, 0, :])

    expected_full = np.array([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
    expected_half = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    assert np.allclose(kx, expected_full)
    assert np.allclose(ky, expected_full)
    assert np.allclose(kz, expected_half)

