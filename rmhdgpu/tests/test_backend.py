from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import rmhdgpu.backend as backend_module
from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config


def test_numpy_backend_selection() -> None:
    config = Config(backend="numpy")
    backend = build_backend(config)
    arr = backend.asarray([1.0, 2.0, 3.0], dtype=config.real_dtype)

    assert backend.xp is np
    assert backend.is_gpu is False
    assert isinstance(backend.to_numpy(arr), np.ndarray)


def test_scalar_to_float_returns_python_float() -> None:
    config = Config(backend="numpy")
    backend = build_backend(config)
    value = backend.asarray(3.5).sum()

    result = backend.scalar_to_float(value)

    assert isinstance(result, float)
    assert result == pytest.approx(3.5)


def test_scipy_cpu_backend_selection_if_available() -> None:
    if importlib.util.find_spec("scipy") is None:
        pytest.skip("SciPy is not installed.")

    config = Config(backend="scipy_cpu", fft_workers=2)
    backend = build_backend(config)
    arr = backend.asarray([1.0, 2.0, 3.0], dtype=config.real_dtype)

    assert backend.xp is np
    assert backend.backend_name == "scipy_cpu"
    assert backend.is_gpu is False
    assert backend.fft_workers == 2
    assert backend.scipy_fft is not None
    assert isinstance(backend.to_numpy(arr), np.ndarray)


def test_scipy_cpu_backend_missing_dependency_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = backend_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "scipy.fft":
            raise ImportError("simulated missing scipy")
        return real_import_module(name)

    monkeypatch.setattr(backend_module.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="backend='scipy_cpu'"):
        build_backend(Config(backend="scipy_cpu"))


def test_cupy_backend_if_available() -> None:
    cupy = pytest.importorskip("cupy")

    config = Config(backend="cupy")
    backend = build_backend(config)
    arr = backend.asarray([1.0, 2.0, 3.0], dtype=config.real_dtype)

    assert backend.xp is cupy
    assert backend.is_gpu is True
    assert isinstance(backend.to_numpy(arr), np.ndarray)
    assert isinstance(backend.scalar_to_float(arr.sum()), float)
