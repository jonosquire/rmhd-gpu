"""Linear eigenmode constructors for the low-beta accretion equation set.

The eigenvector is taken numerically from the equation module's
`linear_matrix(...)`, so the initializer and the tests share one linear
representation and no branch relations are hand coded.

A single stored Fourier mode has all perpendicular gradients parallel to the
same `k_perp`, so every Poisson bracket vanishes identically and the mode
evolves exactly as `exp(eigenvalue * t)` even with the nonlinear terms active.
That is what makes these states useful as exact linear-behaviour tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rmhdgpu.equations import get_equation_module
from rmhdgpu.state import State


#: Selectable eigenvalue branches. `entropy` picks the marginal entropy mode,
#: which has eigenvalue exactly zero when the background entropy gradient
#: vanishes and is the slowest branch otherwise.
EIGENMODE_BRANCHES = (
    "fastest_growing",
    "fastest_decaying",
    "highest_frequency",
    "lowest_frequency",
    "entropy",
)


def _select_mode_index(eigenvalues: np.ndarray, mode: str) -> int:
    if mode == "fastest_growing":
        return int(np.argmax(eigenvalues.real))
    if mode == "fastest_decaying":
        return int(np.argmin(eigenvalues.real))
    if mode == "highest_frequency":
        return int(np.argmax(eigenvalues.imag))
    if mode == "lowest_frequency":
        return int(np.argmin(eigenvalues.imag))
    if mode == "entropy":
        return int(np.argmin(np.abs(eigenvalues)))
    raise ValueError(
        f"Unknown low_beta_accretion eigenmode branch {mode!r}; "
        f"expected one of {list(EIGENMODE_BRANCHES)}."
    )


def low_beta_accretion_mode_eigenvalue(
    grid: Any,
    backend: Any,
    k_indices: Sequence[int],
    mode: str,
    params: Any,
) -> complex:
    """Return the eigenvalue selected by `mode` at the given stored mode index."""

    kx, ky, kz = mode_wavenumbers(grid, backend, k_indices)
    equation_module = get_equation_module("low_beta_accretion")
    eigenvalues = np.linalg.eigvals(equation_module.linear_matrix(kx, ky, kz, params))
    return complex(eigenvalues[_select_mode_index(eigenvalues, mode)])


def mode_wavenumbers(
    grid: Any,
    backend: Any,
    k_indices: Sequence[int],
) -> tuple[float, float, float]:
    """Return the physical `(kx, ky, kz)` for a stored rfftn mode index."""

    if len(k_indices) != 3:
        raise ValueError(f"k_indices must have length 3; got {k_indices!r}.")
    ix, iy, iz = (int(k_indices[0]), int(k_indices[1]), int(k_indices[2]))
    if iz < 0 or iz > grid.Nz // 2:
        raise ValueError(f"k_indices[2] must satisfy 0 <= kz <= Nz//2; got {iz}.")
    return (
        backend.scalar_to_float(grid.kx[ix % grid.Nx, 0, 0]),
        backend.scalar_to_float(grid.ky[0, iy % grid.Ny, 0]),
        backend.scalar_to_float(grid.kz[0, 0, iz]),
    )


def low_beta_accretion_mode_state(
    grid: Any,
    backend: Any,
    field_names: Sequence[str] | None,
    k_indices: Sequence[int],
    amplitude: complex | float = 1.0,
    mode: str = "fastest_growing",
    params: Any | None = None,
) -> State:
    """Return a single low-beta accretion linear eigenmode.

    The state is rescaled so that the equation module's (positive definite)
    `total_energy(...)` equals `amplitude**2`.
    """

    if params is None:
        raise ValueError(
            "low_beta_accretion_mode_state requires params so the background "
            "parameters are defined."
        )

    equation_module = get_equation_module("low_beta_accretion")
    names = list(equation_module.FIELD_NAMES if field_names is None else field_names)
    state = State(grid, backend, field_names=names)

    kx, ky, kz = mode_wavenumbers(grid, backend, k_indices)
    ix = int(k_indices[0]) % grid.Nx
    iy = int(k_indices[1]) % grid.Ny
    iz = int(k_indices[2])

    matrix = equation_module.linear_matrix(kx=kx, ky=ky, kz=kz, params=params)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    vector = eigenvectors[:, _select_mode_index(eigenvalues, mode)]
    scale = np.max(np.abs(vector))
    if scale <= 0.0:
        raise ValueError("Selected low_beta_accretion eigenvector has zero amplitude.")
    vector = vector / scale

    for component, field_name in enumerate(equation_module.FIELD_NAMES):
        state[field_name][ix, iy, iz] = vector[component]

    energy = equation_module.total_energy(state, grid, backend, params)
    if energy <= 0.0:
        raise ValueError(
            "Selected low_beta_accretion eigenvector has zero free energy; "
            "choose a different mode index or branch."
        )
    scale_factor = float(np.abs(amplitude)) / np.sqrt(energy)
    for field_name in state.field_names:
        state[field_name][...] *= scale_factor
    return state
