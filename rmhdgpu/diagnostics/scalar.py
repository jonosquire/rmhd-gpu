"""Simple scalar diagnostics available before a full solver exists."""

from __future__ import annotations

from typing import Any


def compute_scalar_diagnostics(state: Any, grid: Any, fft: Any, backend: Any) -> dict[str, float]:
    """Compute basic real-space scalar diagnostics for each field.

    Each Fourier field is inverse transformed to real space, then the following
    quantities are reported:

    - mean
    - RMS
    - maximum absolute value
    """

    xp = backend.xp
    diagnostics: dict[str, float] = {}

    for name in state.field_names:
        field_real = fft.c2r(state[name])
        diagnostics[f"{name}_mean"] = backend.scalar_to_float(xp.mean(field_real))
        diagnostics[f"{name}_rms"] = backend.scalar_to_float(xp.sqrt(xp.mean(field_real**2)))
        diagnostics[f"{name}_max_abs"] = backend.scalar_to_float(xp.max(xp.abs(field_real)))

    return diagnostics

