from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.operators import dx, dy, dz, inv_lap_perp, lap_perp, poisson_bracket
from rmhdgpu.workspace import Workspace


def _build_operator_fixture():
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    return backend, grid, fft, workspace


def _mesh(grid, backend):
    x = backend.to_numpy(grid.x)[:, None, None]
    y = backend.to_numpy(grid.y)[None, :, None]
    z = backend.to_numpy(grid.z)[None, None, :]
    return x, y, z


def test_derivatives_match_known_trigonometric_mode() -> None:
    backend, grid, fft, _ = _build_operator_fixture()
    x, y, z = _mesh(grid, backend)
    nx, ny, nz = 1, 2, 1
    theta = nx * x + ny * y + nz * z
    field = np.sin(theta)
    field_hat = fft.r2c(backend.asarray(field, dtype=grid.real_dtype))

    dx_real = backend.to_numpy(fft.c2r(dx(field_hat, grid)))
    dy_real = backend.to_numpy(fft.c2r(dy(field_hat, grid)))
    dz_real = backend.to_numpy(fft.c2r(dz(field_hat, grid)))

    expected = np.cos(theta)
    assert np.allclose(dx_real, nx * expected, atol=1.0e-12, rtol=1.0e-12)
    assert np.allclose(dy_real, ny * expected, atol=1.0e-12, rtol=1.0e-12)
    assert np.allclose(dz_real, nz * expected, atol=1.0e-12, rtol=1.0e-12)


def test_lap_perp_matches_analytic_result() -> None:
    backend, grid, fft, _ = _build_operator_fixture()
    x, y, z = _mesh(grid, backend)
    nx, ny, nz = 1, 2, 1
    theta = nx * x + ny * y + nz * z
    field = np.sin(theta)
    field_hat = fft.r2c(backend.asarray(field, dtype=grid.real_dtype))

    lap_real = backend.to_numpy(fft.c2r(lap_perp(field_hat, grid)))

    assert np.allclose(
        lap_real,
        -(nx**2 + ny**2) * field,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_inv_lap_perp_inverts_lap_perp_away_from_kperp0() -> None:
    backend, grid, fft, _ = _build_operator_fixture()
    x, y, z = _mesh(grid, backend)
    field = np.sin(x + 2.0 * y + z)
    field_hat = fft.r2c(backend.asarray(field, dtype=grid.real_dtype))

    recovered_hat = inv_lap_perp(lap_perp(field_hat, grid), grid)

    assert np.allclose(
        backend.to_numpy(recovered_hat),
        backend.to_numpy(field_hat),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_poisson_bracket_of_field_with_itself_is_zero() -> None:
    backend, grid, fft, workspace = _build_operator_fixture()
    x, y, z = _mesh(grid, backend)
    field = np.sin(x + 2.0 * y + z)
    field_hat = fft.r2c(backend.asarray(field, dtype=grid.real_dtype))

    bracket_hat = poisson_bracket(field_hat, field_hat, grid, fft, workspace)
    bracket_real = backend.to_numpy(fft.c2r(bracket_hat))

    assert np.allclose(bracket_real, 0.0, atol=1.0e-12, rtol=1.0e-12)


def test_poisson_bracket_matches_simple_analytic_case() -> None:
    backend, grid, fft, workspace = _build_operator_fixture()
    x, y, _ = _mesh(grid, backend)
    ones_y = np.ones((1, grid.Ny, 1))
    ones_z = np.ones((1, 1, grid.Nz))
    ones_x = np.ones((grid.Nx, 1, 1))
    nx, ny = 2, 1
    f = np.sin(nx * x) * ones_y * ones_z
    g = ones_x * np.sin(ny * y) * ones_z
    f_hat = fft.r2c(backend.asarray(f, dtype=grid.real_dtype))
    g_hat = fft.r2c(backend.asarray(g, dtype=grid.real_dtype))

    bracket_real = backend.to_numpy(fft.c2r(poisson_bracket(f_hat, g_hat, grid, fft, workspace)))
    expected = nx * ny * np.cos(nx * x) * np.cos(ny * y) * ones_z

    assert np.allclose(bracket_real, expected, atol=1.0e-12, rtol=1.0e-12)


def test_parseval_consistency_for_rfft_storage() -> None:
    backend, grid, fft, _ = _build_operator_fixture()
    rng = np.random.default_rng(2024)
    field = rng.standard_normal(grid.real_shape)
    field_hat = backend.to_numpy(fft.r2c(backend.asarray(field, dtype=grid.real_dtype)))

    weights = np.ones(grid.fourier_shape[-1])
    weights[1:-1] = 2.0

    real_norm = np.sum(np.abs(field) ** 2)
    spectral_norm = np.sum(np.abs(field_hat) ** 2 * weights[None, None, :])
    normalization = np.prod(grid.real_shape)

    assert np.allclose(
        real_norm,
        spectral_norm / normalization,
        atol=1.0e-10,
        rtol=1.0e-10,
    ), "Parseval should hold with doubled interior z-modes for one-sided rFFT storage."
