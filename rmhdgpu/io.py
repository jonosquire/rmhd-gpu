"""Minimal file output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy_array(arr: Any, backend: Any | None = None) -> np.ndarray:
    if backend is not None:
        return backend.to_numpy(arr)
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def write_scalar_json(path: str | Path, diagnostics_dict: dict[str, float]) -> None:
    """Write scalar diagnostics to a JSON file."""

    output_path = Path(path)
    payload = {key: float(value) for key, value in diagnostics_dict.items()}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_npz(path: str | Path, backend: Any | None = None, **arrays: Any) -> None:
    """Write arrays to an `.npz` file, converting GPU arrays to NumPy first."""

    output_path = Path(path)
    numpy_arrays = {
        name: _to_numpy_array(array, backend=backend) for name, array in arrays.items()
    }
    np.savez(output_path, **numpy_arrays)

