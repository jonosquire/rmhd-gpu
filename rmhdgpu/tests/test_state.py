from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State


def test_state_fields_created_and_names_preserved() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    state = State(grid, backend, field_names=config.field_names)

    assert state.field_names == config.field_names
    for name in config.field_names:
        assert state[name].shape == grid.fourier_shape
        assert state[name].dtype == grid.complex_dtype


def test_state_copy_is_deep() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    state = State(grid, backend, field_names=config.field_names)
    state["psi"][1, 1, 1] = 3.0 + 4.0j

    copied = state.copy()
    copied["psi"][1, 1, 1] = 0.0

    assert state["psi"][1, 1, 1] == 3.0 + 4.0j


def test_zeros_like_returns_zero_state() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    state = State(grid, backend, field_names=config.field_names)
    state["psi"][1, 1, 1] = 7.0

    zero_state = state.zeros_like()

    assert zero_state.field_names == state.field_names
    assert np.allclose(backend.to_numpy(zero_state["psi"]), 0.0)


def test_apply_mask_updates_all_fields() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    mask = build_dealias_mask(grid, backend)
    state = State(grid, backend, field_names=config.field_names)

    for name in state.field_names:
        state[name][...] = 1.0 + 0.0j

    state.apply_mask(mask)
    psi = backend.to_numpy(state["psi"])

    assert np.allclose(psi, backend.to_numpy(mask).astype(grid.complex_dtype))

