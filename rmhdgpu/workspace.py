"""Reusable scratch buffers for hot operator routines."""

from __future__ import annotations

from typing import Any


class Workspace:
    """Preallocate named real and complex scratch buffers."""

    def __init__(
        self,
        grid: Any,
        backend: Any,
        n_real_buffers: int = 6,
        n_complex_buffers: int = 4,
    ) -> None:
        self.grid = grid
        self.backend = backend
        self.real = {
            f"r{i}": backend.zeros(grid.real_shape, dtype=grid.real_dtype)
            for i in range(n_real_buffers)
        }
        self.complex = {
            f"c{i}": backend.zeros(grid.fourier_shape, dtype=grid.complex_dtype)
            for i in range(n_complex_buffers)
        }

