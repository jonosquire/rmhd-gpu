from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import lap_perp
from rmhdgpu.state import State
from rmhdgpu.steppers import compute_cfl_timestep, evolve_until
from rmhdgpu.workspace import Workspace


def _build_context(config: Config) -> tuple[object, object, FFTManager, Workspace, object]:
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return backend, grid, fft, workspace, mask


def _deterministic_state(backend: object, grid: object, fft: FFTManager, mask: object) -> State:
    state = State(grid, backend, field_names=s09.FIELD_NAMES)
    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)

    phi_real = (
        0.35 * xp.cos(x + z)
        + 0.25 * xp.sin(2.0 * y + z)
    ).astype(grid.real_dtype, copy=False)
    psi_real = (
        0.30 * xp.cos(x + y + z)
        + 0.20 * xp.sin(2.0 * x + y + 2.0 * z)
    ).astype(grid.real_dtype, copy=False)
    upar_real = (0.05 * xp.cos(x + 2.0 * z)).astype(grid.real_dtype, copy=False)
    dbpar_real = (0.04 * xp.sin(y + z)).astype(grid.real_dtype, copy=False)

    phi_hat = fft.r2c(phi_real)
    psi_hat = fft.r2c(psi_real)
    state["psi"][...] = psi_hat * mask
    state["omega"][...] = lap_perp(phi_hat * mask, grid)
    state["upar"][...] = fft.r2c(upar_real) * mask
    state["dbpar"][...] = fft.r2c(dbpar_real) * mask
    return state


def _scale_state(state: State, factor: float) -> State:
    out = state.copy()
    for name in out.field_names:
        out[name][...] *= factor
    return out


def _state_relative_error(a: State, b: State) -> float:
    numerator = 0.0
    denominator = 0.0
    for name in a.field_names:
        a_np = a.backend.to_numpy(a[name])
        b_np = b.backend.to_numpy(b[name])
        numerator += float(np.linalg.norm(a_np - b_np) ** 2)
        denominator += float(np.linalg.norm(b_np) ** 2)
    return np.sqrt(numerator / denominator)


def _build_dissipative_config(cfl_number: float) -> Config:
    config = Config(
        Nx=12,
        Ny=12,
        Nz=12,
        backend="numpy",
        cfl_number=cfl_number,
        dt_init=5.0e-3,
        dt_min=5.0e-4,
        dt_max=2.0e-2,
        use_variable_dt=True,
    )
    for name in config.field_names:
        config.dissipation[name]["nu_perp"] = 3.0e-3
        config.dissipation[name]["nu_par"] = 8.0e-3
        config.dissipation[name]["n_perp"] = 2
        config.dissipation[name]["n_par"] = 1
    return config


def test_cfl_timestep_positive_and_bounded() -> None:
    config = _build_dissipative_config(cfl_number=0.3)
    backend, grid, fft, _, mask = _build_context(config)
    state = _deterministic_state(backend, grid, fft, mask)

    dt = compute_cfl_timestep(state, grid, fft, config)

    assert dt > 0.0
    assert dt >= config.dt_min
    assert dt <= config.dt_max


def test_larger_amplitude_gives_smaller_dt() -> None:
    config = _build_dissipative_config(cfl_number=0.3)
    config.dt_max = None
    backend, grid, fft, _, mask = _build_context(config)
    state = _deterministic_state(backend, grid, fft, mask)
    state_scaled = _scale_state(state, 4.0)

    dt_base = compute_cfl_timestep(state, grid, fft, config)
    dt_scaled = compute_cfl_timestep(state_scaled, grid, fft, config)

    assert dt_scaled < dt_base, f"Expected larger amplitudes to reduce dt, but got {dt_scaled} >= {dt_base}."


def test_different_cfl_numbers_converge_to_same_solution() -> None:
    config_slow = _build_dissipative_config(cfl_number=0.2)
    config_fast = _build_dissipative_config(cfl_number=0.4)

    backend_slow, grid_slow, fft_slow, workspace_slow, mask_slow = _build_context(config_slow)
    backend_fast, grid_fast, fft_fast, workspace_fast, mask_fast = _build_context(config_fast)

    state_slow = _deterministic_state(backend_slow, grid_slow, fft_slow, mask_slow)
    state_fast = _deterministic_state(backend_fast, grid_fast, fft_fast, mask_fast)

    rhs_kwargs_slow = {
        "grid": grid_slow,
        "fft": fft_slow,
        "workspace": workspace_slow,
        "params": config_slow,
        "dealias_mask": mask_slow,
    }
    rhs_kwargs_fast = {
        "grid": grid_fast,
        "fft": fft_fast,
        "workspace": workspace_fast,
        "params": config_fast,
        "dealias_mask": mask_fast,
    }

    final_slow, _ = evolve_until(
        state_slow,
        5.0e-2,
        s09.ideal_rhs,
        s09.build_dissipation_operators(grid_slow, config_slow),
        rhs_kwargs=rhs_kwargs_slow,
        params=config_slow,
    )
    final_fast, _ = evolve_until(
        state_fast,
        5.0e-2,
        s09.ideal_rhs,
        s09.build_dissipation_operators(grid_fast, config_fast),
        rhs_kwargs=rhs_kwargs_fast,
        params=config_fast,
    )

    relative_error = _state_relative_error(final_fast, final_slow)
    assert relative_error < 5.0e-3, f"CFL refinement mismatch too large: relative_error={relative_error:.3e}"


def test_variable_dt_close_to_fixed_dt_reference() -> None:
    config_var = _build_dissipative_config(cfl_number=0.2)
    config_fixed = _build_dissipative_config(cfl_number=0.2)
    config_fixed.use_variable_dt = False

    backend_var, grid_var, fft_var, workspace_var, mask_var = _build_context(config_var)
    backend_fixed, grid_fixed, fft_fixed, workspace_fixed, mask_fixed = _build_context(config_fixed)

    state_var = _deterministic_state(backend_var, grid_var, fft_var, mask_var)
    state_fixed = _deterministic_state(backend_fixed, grid_fixed, fft_fixed, mask_fixed)

    rhs_kwargs_var = {
        "grid": grid_var,
        "fft": fft_var,
        "workspace": workspace_var,
        "params": config_var,
        "dealias_mask": mask_var,
    }
    rhs_kwargs_fixed = {
        "grid": grid_fixed,
        "fft": fft_fixed,
        "workspace": workspace_fixed,
        "params": config_fixed,
        "dealias_mask": mask_fixed,
    }

    final_var, _ = evolve_until(
        state_var,
        5.0e-2,
        s09.ideal_rhs,
        s09.build_dissipation_operators(grid_var, config_var),
        rhs_kwargs=rhs_kwargs_var,
        params=config_var,
    )
    final_fixed, _ = evolve_until(
        state_fixed,
        5.0e-2,
        s09.ideal_rhs,
        s09.build_dissipation_operators(grid_fixed, config_fixed),
        rhs_kwargs=rhs_kwargs_fixed,
        params=config_fixed,
        fixed_dt=1.0e-3,
    )

    relative_error = _state_relative_error(final_var, final_fixed)
    assert relative_error < 3.0e-3, f"Variable-dt result deviates too much from fixed reference: {relative_error:.3e}"
