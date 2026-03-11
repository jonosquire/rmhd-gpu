from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid


CPU_FFT_BACKENDS = ["numpy"]
if importlib.util.find_spec("scipy") is not None:
    CPU_FFT_BACKENDS.append("scipy_cpu")


def _build_fft_fixture(backend_name: str = "numpy"):
    config = Config(Nx=8, Ny=8, Nz=8, backend=backend_name)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    return config, backend, grid, fft


@pytest.mark.parametrize("backend_name", CPU_FFT_BACKENDS)
def test_random_real_roundtrip(backend_name: str) -> None:
    _, backend, grid, fft = _build_fft_fixture(backend_name)
    rng = np.random.default_rng(123)
    field = backend.asarray(rng.standard_normal(grid.real_shape), dtype=grid.real_dtype)

    recovered = fft.c2r(fft.r2c(field))

    assert np.allclose(
        backend.to_numpy(field),
        backend.to_numpy(recovered),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize("backend_name", CPU_FFT_BACKENDS)
def test_repeated_roundtrip_stability(backend_name: str) -> None:
    _, backend, grid, fft = _build_fft_fixture(backend_name)
    rng = np.random.default_rng(456)
    field = backend.asarray(rng.standard_normal(grid.real_shape), dtype=grid.real_dtype)
    original = backend.to_numpy(field).copy()

    current = field
    for _ in range(5):
        current = fft.c2r(fft.r2c(current))

    assert np.allclose(original, backend.to_numpy(current), atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize("backend_name", CPU_FFT_BACKENDS)
def test_known_cosine_mode_lands_in_expected_coefficient(backend_name: str) -> None:
    _, backend, grid, fft = _build_fft_fixture(backend_name)
    x = backend.to_numpy(grid.x)[:, None, None]
    y = backend.to_numpy(grid.y)[None, :, None]
    z = backend.to_numpy(grid.z)[None, None, :]
    nx, ny, nz = 1, 2, 1
    field = np.cos(nx * x + ny * y + nz * z)

    field_hat = backend.to_numpy(fft.r2c(backend.asarray(field, dtype=grid.real_dtype)))
    max_index = np.unravel_index(np.argmax(np.abs(field_hat)), field_hat.shape)
    expected_amplitude = np.prod(grid.real_shape) / 2.0

    assert max_index == (nx, ny, nz)
    np.testing.assert_allclose(
        np.abs(field_hat[max_index]),
        expected_amplitude,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


@pytest.mark.parametrize("backend_name", CPU_FFT_BACKENDS)
def test_mean_preservation_under_roundtrip(backend_name: str) -> None:
    _, backend, grid, fft = _build_fft_fixture(backend_name)
    rng = np.random.default_rng(789)
    field = rng.standard_normal(grid.real_shape)
    original_mean = field.mean()

    recovered = backend.to_numpy(fft.c2r(fft.r2c(backend.asarray(field, dtype=grid.real_dtype))))

    np.testing.assert_allclose(
        recovered.mean(),
        original_mean,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
