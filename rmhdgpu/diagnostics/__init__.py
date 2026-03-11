"""Diagnostics namespace."""

from rmhdgpu.diagnostics.alfvenic import (
    alfvenic_cross_helicity,
    alfvenic_cross_helicity_rhs_budget,
    alfvenic_energy,
    alfvenic_energy_rhs_budget,
)
from rmhdgpu.diagnostics.scalar import compute_scalar_diagnostics
from rmhdgpu.diagnostics.spectra import perpendicular_energy_spectrum_from_state

__all__ = [
    "alfvenic_cross_helicity",
    "alfvenic_cross_helicity_rhs_budget",
    "alfvenic_energy",
    "alfvenic_energy_rhs_budget",
    "compute_scalar_diagnostics",
    "perpendicular_energy_spectrum_from_state",
]
