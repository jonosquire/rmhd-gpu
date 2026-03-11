"""Equation module namespace."""

from rmhdgpu.equations import s09
from rmhdgpu.equations.s09 import (
    FIELD_NAMES,
    alpha_from_params,
    build_dissipation_operators,
    dissipation_operator,
    derive_j_hat,
    derive_phi_hat,
    ideal_rhs,
    linear_matrix,
    rhs,
)

__all__ = [
    "FIELD_NAMES",
    "alpha_from_params",
    "build_dissipation_operators",
    "dissipation_operator",
    "derive_j_hat",
    "derive_phi_hat",
    "ideal_rhs",
    "linear_matrix",
    "rhs",
    "s09",
]
