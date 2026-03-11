from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import apply_mask, build_dealias_mask


def test_mask_shape_and_basic_retention_rules() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    mask = backend.to_numpy(build_dealias_mask(grid, backend))

    assert mask.shape == grid.fourier_shape
    assert mask[0, 0, 0]
    assert mask[1, 1, 1]
    assert not mask[3, 0, 0]
    assert not mask[0, 3, 0]
    assert not mask[0, 0, 3]


def test_apply_mask_preserves_shape_and_dtype() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    mask = build_dealias_mask(grid, backend)
    field_hat = backend.ones(grid.fourier_shape, dtype=grid.complex_dtype)

    result = apply_mask(field_hat, mask)

    assert result.shape == grid.fourier_shape
    assert result.dtype == grid.complex_dtype


def test_mode_count_matches_tensor_product_two_thirds_rule() -> None:
    config = Config(Nx=6, Ny=6, Nz=6)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    mask = backend.to_numpy(build_dealias_mask(grid, backend))

    expected_count = 3 * 3 * 2
    assert int(mask.sum()) == expected_count

