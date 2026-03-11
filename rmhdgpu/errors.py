"""Custom exceptions used by rmhdgpu."""

from __future__ import annotations


class NonFiniteStateError(RuntimeError):
    """Raised when a state contains NaN or infinite values during a run."""
