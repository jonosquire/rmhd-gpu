from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.grid import build_grid
from rmhdgpu.state import State
from rmhdgpu.steppers import ssprk3_step


def _one_field_state() -> tuple[Config, object, object, State]:
    config = Config(Nx=4, Ny=4, Nz=4, backend="numpy", field_names=["y"])
    backend = build_backend(config)
    grid = build_grid(config, backend)
    state = State(grid, backend, field_names=config.field_names)
    return config, backend, grid, state


def test_ssprk3_zero_rhs() -> None:
    _, backend, _, state = _one_field_state()
    rng = np.random.default_rng(123)
    state["y"][...] = rng.standard_normal(state["y"].shape) + 1j * rng.standard_normal(
        state["y"].shape
    )
    initial = backend.to_numpy(state["y"]).copy()

    def zero_rhs(current_state: State) -> State:
        return current_state.zeros_like()

    stepped = ssprk3_step(state, 0.1, zero_rhs)

    np.testing.assert_allclose(backend.to_numpy(stepped["y"]), initial, atol=1.0e-15, rtol=1.0e-15)


def test_ssprk3_simple_linear_scalar_ode() -> None:
    _, backend, _, state = _one_field_state()
    rng = np.random.default_rng(456)
    state["y"][...] = rng.standard_normal(state["y"].shape) + 1j * rng.standard_normal(
        state["y"].shape
    )

    lam = -0.4 + 0.7j
    dt = 0.05
    z = lam * dt
    stability_polynomial = 1.0 + z + 0.5 * z**2 + (z**3) / 6.0
    initial = backend.to_numpy(state["y"]).copy()

    def linear_rhs(current_state: State) -> State:
        out = current_state.zeros_like()
        out["y"][...] = lam * current_state["y"]
        return out

    stepped = ssprk3_step(state, dt, linear_rhs)

    np.testing.assert_allclose(
        backend.to_numpy(stepped["y"]),
        stability_polynomial * initial,
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_ssprk3_preserves_shapes_and_dtypes() -> None:
    config = Config(Nx=4, Ny=4, Nz=4, backend="numpy")
    backend = build_backend(config)
    grid = build_grid(config, backend)
    state = State(grid, backend, field_names=config.field_names)
    rng = np.random.default_rng(789)

    for name in state.field_names:
        state[name][...] = rng.standard_normal(state[name].shape) + 1j * rng.standard_normal(
            state[name].shape
        )

    def linear_rhs(current_state: State) -> State:
        out = current_state.zeros_like()
        for field_name in out.field_names:
            out[field_name][...] = (1.0 - 0.5j) * current_state[field_name]
        return out

    stepped = ssprk3_step(state, 0.02, linear_rhs)

    assert stepped.field_names == state.field_names
    for name in stepped.field_names:
        assert stepped[name].shape == grid.fourier_shape
        assert stepped[name].dtype == grid.complex_dtype

