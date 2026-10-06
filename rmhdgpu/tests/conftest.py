from __future__ import annotations

import sys
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(params=("numpy", "cupy"))
def production_backend(request):
    """Exercise controlled forcing on CPU and a real GPU when available."""
    if request.param == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"no usable CUDA device: {exc}")
        if count == 0:
            pytest.skip("no CUDA device")
    return request.param
