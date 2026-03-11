"""Compatibility shim for the real homogeneous Schekochihin-2009 prototype."""

from rmhdgpu.equations.s09 import (
    FIELD_NAMES,
    alpha_from_params,
    derive_j_hat,
    derive_phi_hat,
    linear_matrix,
    rhs,
)

__all__ = [
    "FIELD_NAMES",
    "alpha_from_params",
    "derive_j_hat",
    "derive_phi_hat",
    "linear_matrix",
    "rhs",
]
