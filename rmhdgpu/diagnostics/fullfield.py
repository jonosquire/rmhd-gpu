"""Minimal helpers for extracting full real-space fields."""

from __future__ import annotations

from typing import Any, Iterable


def extract_full_fields(
    state: Any,
    fft: Any,
    backend: Any,
    field_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Inverse transform requested fields and return NumPy arrays."""

    names = list(field_names) if field_names is not None else state.field_names
    return {name: backend.to_numpy(fft.c2r(state[name])) for name in names}

