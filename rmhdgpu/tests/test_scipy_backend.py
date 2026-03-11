from __future__ import annotations

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid


scipy_fft = pytest.importorskip("scipy.fft")


def _build_fft_objects(backend_name: str, fft_workers: int | None = None):
    config = Config(Nx=8, Ny=8, Nz=8, backend=backend_name, fft_workers=fft_workers)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    return config, backend, grid, fft


def test_build_scipy_backend() -> None:
    config = Config(backend="scipy_cpu", fft_workers=2)
    backend = build_backend(config)

    assert backend.xp is np
    assert backend.backend_name == "scipy_cpu"
    assert backend.is_gpu is False
    assert backend.fft_workers == 2
    assert backend.scipy_fft is scipy_fft


def test_scipy_fft_roundtrip_matches_numpy() -> None:
    _, backend_np, grid_np, fft_np = _build_fft_objects("numpy")
    _, backend_sp, grid_sp, fft_sp = _build_fft_objects("scipy_cpu", fft_workers=2)

    rng = np.random.default_rng(2468)
    field = rng.standard_normal(grid_np.real_shape)

    field_np = backend_np.asarray(field, dtype=grid_np.real_dtype)
    field_sp = backend_sp.asarray(field, dtype=grid_sp.real_dtype)

    hat_np = backend_np.to_numpy(fft_np.r2c(field_np))
    hat_sp = backend_sp.to_numpy(fft_sp.r2c(field_sp))
    recovered_np = backend_np.to_numpy(fft_np.c2r(fft_np.r2c(field_np)))
    recovered_sp = backend_sp.to_numpy(fft_sp.c2r(fft_sp.r2c(field_sp)))

    np.testing.assert_allclose(hat_sp, hat_np, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(recovered_sp, recovered_np, atol=1.0e-12, rtol=1.0e-12)


def test_scipy_backend_accepts_workers() -> None:
    _, backend, grid, fft = _build_fft_objects("scipy_cpu", fft_workers=1)
    rng = np.random.default_rng(1357)
    field = backend.asarray(rng.standard_normal(grid.real_shape), dtype=grid.real_dtype)

    transformed = fft.r2c(field)
    recovered = fft.c2r(transformed)

    assert backend.fft_workers == 1
    np.testing.assert_allclose(
        backend.to_numpy(recovered),
        backend.to_numpy(field),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
