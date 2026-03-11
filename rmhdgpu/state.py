"""Simple container for named Fourier-space fields."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from rmhdgpu.masks import apply_mask


class State:
    """Store a set of Fourier fields with shared shape and dtype."""

    def __init__(
        self,
        grid: Any,
        backend: Any,
        field_names: Iterable[str] | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        self.grid = grid
        self.backend = backend

        if data is not None:
            names = list(data.keys()) if field_names is None else list(field_names)
            self._fields = {}
            for name in names:
                if name not in data:
                    raise ValueError(f"Missing field {name!r} in provided data.")
                arr = backend.asarray(data[name], dtype=grid.complex_dtype)
                self._validate_field(name, arr)
                self._fields[name] = arr.copy()
        else:
            names = list(field_names or [])
            if not names:
                raise ValueError("State requires field_names or data.")
            self._fields = {
                name: backend.zeros(grid.fourier_shape, dtype=grid.complex_dtype)
                for name in names
            }

    @property
    def field_names(self) -> list[str]:
        return list(self._fields.keys())

    def _validate_field(self, name: str, arr: Any) -> None:
        if arr.shape != self.grid.fourier_shape:
            raise ValueError(
                f"Field {name!r} has shape {arr.shape}, expected {self.grid.fourier_shape}."
            )
        if arr.dtype != self.grid.complex_dtype:
            raise ValueError(
                f"Field {name!r} has dtype {arr.dtype}, expected {self.grid.complex_dtype}."
            )

    def __getitem__(self, name: str) -> Any:
        return self._fields[name]

    def __setitem__(self, name: str, value: Any) -> None:
        arr = self.backend.asarray(value, dtype=self.grid.complex_dtype)
        self._validate_field(name, arr)
        self._fields[name][...] = arr

    def items(self):
        return self._fields.items()

    def copy(self) -> "State":
        return State(self.grid, self.backend, data=self._fields)

    def zeros_like(self) -> "State":
        return State(self.grid, self.backend, field_names=self.field_names)

    def apply_mask(self, mask: Any) -> "State":
        for field in self._fields.values():
            apply_mask(field, mask)
        return self

