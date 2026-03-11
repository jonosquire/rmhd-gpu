from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.eigenmodes import entropy_mode_state
from rmhdgpu.initconds.random_modes import random_band_limited_field
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace


def _build_context() -> tuple[Config, object, object, FFTManager, Workspace, object]:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy", vA=1.7, cs2_over_vA2=0.5)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, workspace, mask


def test_rhs_zero_state() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state = State(grid, backend, field_names=s09.FIELD_NAMES)

    rhs_state = s09.rhs(state, grid, fft, workspace, config, dealias_mask=mask)

    for name in rhs_state.field_names:
        assert np.allclose(
            backend.to_numpy(rhs_state[name]),
            0.0,
            atol=0.0,
            rtol=0.0,
        ), f"Expected exact zero RHS for field {name}."


def test_rhs_shapes_and_field_names() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state = State(grid, backend, field_names=s09.FIELD_NAMES)

    for offset, name in enumerate(state.field_names):
        state[name][...] = random_band_limited_field(
            grid=grid,
            backend=backend,
            fft=fft,
            kmin=1.0,
            kmax=3.0,
            seed=100 + offset,
            rms=0.2,
            dealias_mask=mask,
        )

    rhs_state = s09.rhs(state, grid, fft, workspace, config, dealias_mask=mask)

    assert rhs_state.field_names == s09.FIELD_NAMES
    for name in rhs_state.field_names:
        assert rhs_state[name].shape == grid.fourier_shape
        assert rhs_state[name].dtype == grid.complex_dtype


def test_linear_matrix_eigenvalues() -> None:
    config, _, _, _, _, _ = _build_context()
    alpha = s09.alpha_from_params(config)
    kz = 1.25

    matrix = s09.linear_matrix(kx=1.0, ky=2.0, kz=kz, params=config)
    eigenvalues = np.sort_complex(np.linalg.eigvals(matrix))
    expected = np.sort_complex(
        np.array(
            [
                1j * config.vA * kz,
                -1j * config.vA * kz,
                1j * config.vA * np.sqrt(alpha) * kz,
                -1j * config.vA * np.sqrt(alpha) * kz,
                0.0,
            ],
            dtype=np.complex128,
        )
    )

    np.testing.assert_allclose(eigenvalues, expected, atol=1.0e-12, rtol=1.0e-12)


def test_entropy_rhs_when_only_s_present() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state = entropy_mode_state(
        grid=grid,
        backend=backend,
        field_names=s09.FIELD_NAMES,
        k_indices=(1, 1, 1),
        amplitude=1.0,
    )

    rhs_state = s09.rhs(state, grid, fft, workspace, config, dealias_mask=mask)

    for name in rhs_state.field_names:
        assert np.allclose(
            backend.to_numpy(rhs_state[name]),
            0.0,
            atol=1.0e-14,
            rtol=1.0e-14,
        ), f"Expected stationary pure-entropy state, but RHS[{name}] was nonzero."
