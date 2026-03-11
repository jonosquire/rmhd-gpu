"""Time integration helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from rmhdgpu.forcing import apply_forcing_kick, generate_forcing_kick
from rmhdgpu.state import State
from rmhdgpu.utils import check_state_finite


def state_linear_combination(
    template: State,
    terms: Iterable[tuple[complex | float, State]],
) -> State:
    """Return a linear combination of states using `template` for metadata."""

    result = template.zeros_like()
    for name in result.field_names:
        out = result[name]
        out[...] = 0.0
        for coefficient, state in terms:
            out[...] += coefficient * state[name]
    return result


def ssprk3_step(
    state: State,
    dt: float,
    rhs_func: Any,
    rhs_kwargs: dict[str, Any] | None = None,
) -> State:
    """Advance one step with the classical three-stage SSPRK3 scheme.

    The Shu-Osher form used here is

    `u1 = u^n + dt L(u^n)`

    `u2 = 3/4 u^n + 1/4 (u1 + dt L(u1))`

    `u^{n+1} = 1/3 u^n + 2/3 (u2 + dt L(u2))`

    The input state is not modified.
    """

    kwargs = {} if rhs_kwargs is None else dict(rhs_kwargs)

    rhs0 = rhs_func(state, **kwargs)
    stage1 = state_linear_combination(state, [(1.0, state), (dt, rhs0)])

    rhs1 = rhs_func(stage1, **kwargs)
    predictor2 = state_linear_combination(stage1, [(1.0, stage1), (dt, rhs1)])
    stage2 = state_linear_combination(state, [(0.75, state), (0.25, predictor2)])

    rhs2 = rhs_func(stage2, **kwargs)
    predictor3 = state_linear_combination(stage2, [(1.0, stage2), (dt, rhs2)])
    return state_linear_combination(state, [(1.0 / 3.0, state), (2.0 / 3.0, predictor3)])


def state_fieldwise_multiply(state: State, factors: dict[str, Any]) -> State:
    """Return a new state with each field multiplied by its matching factor."""

    result = state.zeros_like()
    for name in result.field_names:
        result[name][...] = factors[name] * state[name]
    return result


def _build_exponential_factors(
    state: State,
    linear_ops: dict[str, Any],
    dt: float,
    fraction: float,
    sign: float,
) -> dict[str, Any]:
    xp = state.backend.xp
    return {
        name: xp.exp(sign * fraction * dt * linear_ops[name])
        for name in state.field_names
    }


def compute_cfl_timestep(
    state: State,
    grid: Any,
    fft: Any,
    params: Any,
    dt_prev: float | None = None,
) -> float:
    """Estimate a CFL-limited timestep from perpendicular and parallel speeds.

    The estimate uses

    - perpendicular advection by `u_perp = zhat x grad_perp phi`
    - field-line advection by `b_perp = zhat x grad_perp psi`
    - parallel linear propagation at speeds `vA` and `c_slow = vA * sqrt(alpha)`

    The returned timestep is `cfl_number * min(candidates)` clipped by
    `dt_min` and `dt_max` when those bounds are configured.
    """

    from rmhdgpu.equations.s09 import alpha_from_params, derive_phi_hat
    from rmhdgpu.operators import dx, dy

    backend = state.backend
    xp = backend.xp

    phi_hat = derive_phi_hat(state["omega"], grid)
    psi_hat = state["psi"]

    ux = -fft.c2r(dy(phi_hat, grid))
    uy = fft.c2r(dx(phi_hat, grid))
    bx = -fft.c2r(dy(psi_hat, grid))
    by = fft.c2r(dx(psi_hat, grid))

    def _speed_max(field: Any) -> float:
        return backend.scalar_to_float(xp.max(xp.abs(field)))

    def _safe_dt(length: float, speed: float) -> float:
        return math.inf if speed <= 0.0 else length / speed

    candidates = [
        _safe_dt(grid.dx, _speed_max(ux)),
        _safe_dt(grid.dy, _speed_max(uy)),
        _safe_dt(grid.dx, _speed_max(bx)),
        _safe_dt(grid.dy, _speed_max(by)),
    ]

    vA = float(getattr(params, "vA"))
    alpha = alpha_from_params(params)
    c_slow = vA * math.sqrt(alpha)

    candidates.append(math.inf if vA <= 0.0 else grid.dz / vA)
    candidates.append(math.inf if c_slow <= 0.0 else grid.dz / c_slow)

    finite_candidates = [value for value in candidates if math.isfinite(value) and value > 0.0]
    if finite_candidates:
        dt = float(getattr(params, "cfl_number")) * min(finite_candidates)
    else:
        dt = float(getattr(params, "dt_init"))

    if getattr(params, "dt_max") is not None:
        dt = min(dt, float(getattr(params, "dt_max")))
    if getattr(params, "dt_min") is not None:
        dt = max(dt, float(getattr(params, "dt_min")))

    return dt


def if_ssprk3_step(
    state: State,
    dt: float,
    ideal_rhs_func: Any,
    linear_ops: dict[str, Any],
    rhs_kwargs: dict[str, Any] | None = None,
) -> State:
    """Advance one step with an integrating-factor RK3 scheme.

    The method applies the standard third-order SSPRK / Kutta tableau to the
    transformed variable `v = exp(+D t) q`, where the physical state obeys

    `q_t = -D q + N(q)`

    with diagonal nonnegative damping operators `D_i(k)` for each field.

    Over one step of size `dt`, with local stage times `c = [0, 1, 1/2]`, the
    transformed stages are

    `k1 = N(q_n)`

    `q1 = E(dt) * (q_n + dt * k1)`

    `k2 = E(-dt) * N(q1)`

    `q2 = E(dt/2) * (q_n + dt/4 * (k1 + k2))`

    `k3 = E(-dt/2) * N(q2)`

    `q_{n+1} = E(dt) * (q_n + dt * (k1/6 + k2/6 + 2 k3/3))`

    where `E(c dt) = exp(-D * c dt)` acts fieldwise in Fourier space.

    When all `D_i = 0`, this reduces to the existing ideal RK3 step.
    """

    kwargs = {} if rhs_kwargs is None else dict(rhs_kwargs)

    exp_full = _build_exponential_factors(state, linear_ops, dt, 1.0, sign=-1.0)
    exp_half = _build_exponential_factors(state, linear_ops, dt, 0.5, sign=-1.0)
    exp_inv_full = _build_exponential_factors(state, linear_ops, dt, 1.0, sign=1.0)
    exp_inv_half = _build_exponential_factors(state, linear_ops, dt, 0.5, sign=1.0)

    k1 = ideal_rhs_func(state, **kwargs)
    stage1_unscaled = state_linear_combination(state, [(1.0, state), (dt, k1)])
    stage1 = state_fieldwise_multiply(stage1_unscaled, exp_full)

    n2 = ideal_rhs_func(stage1, **kwargs)
    k2 = state_fieldwise_multiply(n2, exp_inv_full)
    stage2_unscaled = state_linear_combination(
        state,
        [(1.0, state), (0.25 * dt, k1), (0.25 * dt, k2)],
    )
    stage2 = state_fieldwise_multiply(stage2_unscaled, exp_half)

    n3 = ideal_rhs_func(stage2, **kwargs)
    k3 = state_fieldwise_multiply(n3, exp_inv_half)
    final_unscaled = state_linear_combination(
        state,
        [(1.0, state), (dt / 6.0, k1), (dt / 6.0, k2), (2.0 * dt / 3.0, k3)],
    )
    return state_fieldwise_multiply(final_unscaled, exp_full)


def evolve_until(
    state: State,
    t_final: float,
    ideal_rhs_func: Any,
    linear_ops: dict[str, Any],
    rhs_kwargs: dict[str, Any] | None = None,
    params: Any | None = None,
    fixed_dt: float | None = None,
    check_every: int | None = None,
    stepper_func: Any | None = None,
    forcing_rng: np.random.Generator | None = None,
) -> tuple[State, dict[str, float | int]]:
    """Advance until `t_final`, using variable or fixed timestep selection.

    Runtime non-finite checks are performed at the configured interval when
    `params.fail_on_nonfinite` is true.
    """

    kwargs = {} if rhs_kwargs is None else dict(rhs_kwargs)
    params_obj = params if params is not None else kwargs.get("params")
    if params_obj is None:
        raise ValueError("evolve_until requires params directly or in rhs_kwargs['params'].")

    grid = kwargs.get("grid")
    fft = kwargs.get("fft")
    if grid is None or fft is None:
        raise ValueError("evolve_until requires 'grid' and 'fft' entries in rhs_kwargs.")

    if check_every is None:
        check_every = int(getattr(params_obj, "runtime_check_every", 1))
    if stepper_func is None:
        stepper_func = if_ssprk3_step

    forcing_rng_obj = forcing_rng
    if getattr(params_obj, "use_forcing", False) and forcing_rng_obj is None:
        forcing_rng_obj = np.random.default_rng(getattr(params_obj, "forcing_seed", None))

    current = state
    t = 0.0
    dt_prev: float | None = None
    steps = 0

    if getattr(params_obj, "fail_on_nonfinite", True):
        check_state_finite(current, current.backend, time=t, step=steps, context="time integration startup")

    while t < t_final - 1.0e-15:
        if fixed_dt is not None:
            dt = fixed_dt
        elif getattr(params_obj, "use_variable_dt", True):
            dt = compute_cfl_timestep(current, grid, fft, params_obj, dt_prev=dt_prev)
        else:
            dt = float(getattr(params_obj, "dt_init"))

        dt = min(dt, t_final - t)
        current = stepper_func(current, dt, ideal_rhs_func, linear_ops, rhs_kwargs=kwargs)
        if getattr(params_obj, "use_forcing", False):
            forcing_kick = generate_forcing_kick(
                current,
                grid,
                fft,
                current.backend,
                params_obj,
                forcing_rng_obj,
                dt,
            )
            current = apply_forcing_kick(current, forcing_kick)
        t += dt
        dt_prev = dt
        steps += 1

        if getattr(params_obj, "fail_on_nonfinite", True):
            should_check = (steps % check_every == 0) or (t >= t_final - 1.0e-15)
            if should_check:
                check_state_finite(
                    current,
                    current.backend,
                    time=t,
                    step=steps,
                    context="time integration",
                )

    info: dict[str, float | int] = {
        "t": t,
        "steps": steps,
        "dt_last": 0.0 if dt_prev is None else dt_prev,
    }
    return current, info
