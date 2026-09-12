"""Targeted tests for the `low_beta_accretion` disc RMHD equation set.

The physics checks are anchored on two published results for the
`vA_over_U = 0` limit of this system, which is exactly the rotating RMHD
(RRMHD) model of Kawazura et al. (2022, JPP 88, 905880311):

- their dispersion relation (3.2), and
- their maximum MRI growth rate (3.3),
  `gamma_max/Omega = sqrt(5 beta/18 [20 beta + 15 - sqrt(8(50 beta^2 + 75 beta + 18))])`
  for `q = 3/2`, `Gamma = 5/3` and `beta = 8 pi p_0 / B_0^2 = 2 beta_tilde/gamma`.

Everything else (finite `vA_over_U`, background gradients) is checked for
internal consistency: the coded RHS must equal the coded linear matrix, and the
coded energy-budget source terms must equal the exact time derivative of the
coded free energy along the coded RHS.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import available_equation_sets, get_equation_module, low_beta_accretion as lba
from rmhdgpu.fft import FFTManager
from rmhdgpu.fourier_diagnostics import modal_average
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.initconds.eigenmodes_low_beta_accretion import (
    low_beta_accretion_mode_eigenvalue,
    low_beta_accretion_mode_state,
)
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import inv_lap_perp
from rmhdgpu.run import main
from rmhdgpu.runfile import resolve_run_settings
from rmhdgpu.state import State
from rmhdgpu.steppers import ssprk3_step
from rmhdgpu.workspace import Workspace


DISC_PARAMS = {
    "cs2_over_vA2": 0.6,
    "gamma_ad": 5.0 / 3.0,
    "q_shear": 1.5,
    "vA_over_U": 0.4,
    "B_hat": -0.9,
    "P_hat": -2.7,
    "rho_hat": -1.3,
}


def _config(**overrides) -> Config:
    values = {
        "equation_set": "low_beta_accretion",
        "Nx": 8,
        "Ny": 8,
        "Nz": 8,
        "backend": "numpy",
        **DISC_PARAMS,
    }
    values.update(overrides)
    return Config(**values)


def _context(config: Config):
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend) if config.dealias else None
    return backend, grid, fft, workspace, mask


def _random_state(config, grid, backend, fft, mask, *, seed: int = 7, energy: float = 1.0) -> State:
    return build_initial_state(
        "random_spectrum",
        parameters={"n_min": 1.0, "n_max": 3.0, "alpha": 0.0, "init_energy": energy, "seed": seed},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )


def _exact_energy_rate(state: State, rhs: State, grid, backend, params) -> float:
    """Return `d_t E` as the exact directional derivative of the coded energy.

    This is deliberately written out from the quadratic form in
    `total_energy_modal_density` rather than reusing the budget helpers, so it
    is an independent check of the source terms.
    """

    xp = backend.xp
    p = lba.derived_parameters(params)
    phi_hat = inv_lap_perp(state["omega"], grid)
    phi_t_hat = inv_lap_perp(rhs["omega"], grid)
    sigma_hat = state["dbpar"] / p.beta_tilde + state["drho"]
    sigma_t_hat = rhs["dbpar"] / p.beta_tilde + rhs["drho"]
    density = (
        grid.kperp2 * (xp.conj(phi_hat) * phi_t_hat + xp.conj(state["psi"]) * rhs["psi"])
        + xp.conj(state["upar"]) * rhs["upar"]
        + p.dbpar_energy_weight * xp.conj(state["dbpar"]) * rhs["dbpar"]
        + p.entropy_energy_weight * xp.conj(sigma_hat) * sigma_t_hat
    )
    return modal_average(density, grid, backend)


def _advance(state, *, steps, dt, config, grid, fft, workspace, mask) -> State:
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }
    current = state
    for _ in range(steps):
        current = ssprk3_step(current, dt, lba.ideal_rhs, rhs_kwargs=rhs_kwargs)
    return current


# --------------------------------------------------------------------------
# registration and configuration
# --------------------------------------------------------------------------


def test_equation_set_is_registered() -> None:
    assert "low_beta_accretion" in available_equation_sets()
    assert get_equation_module("low_beta_accretion") is lba
    assert _config().field_names == ["psi", "omega", "upar", "dbpar", "drho"]
    assert lba.DEFAULT_INITIAL_CONDITION == "low_beta_accretion_mode"


def test_derived_parameters_match_the_documented_definitions() -> None:
    config = _config()
    p = lba.derived_parameters(config)
    bt, gam = config.cs2_over_vA2, config.gamma_ad
    assert p.chi == pytest.approx(bt * config.P_hat / gam + config.B_hat + 1.0)
    assert p.C_b == pytest.approx(config.B_hat - 1.0 - config.P_hat / gam)
    assert p.C_rho == pytest.approx(
        config.B_hat - 1.0 + bt * config.P_hat / gam - (1.0 + bt) * config.rho_hat
    )
    assert p.S_hat == pytest.approx(config.P_hat - gam * config.rho_hat)
    assert p.alpha_b + p.alpha_rho == pytest.approx(1.0)
    assert p.dbpar_energy_weight == pytest.approx(1.0 + 1.0 / bt)
    # Normalised parallel speeds: Alfven = 1, slow = cs/sqrt(vA^2 + cs^2).
    speeds = lba.characteristic_speeds(config)
    assert speeds[0] == pytest.approx(1.0)
    assert speeds[1] == pytest.approx(np.sqrt(bt / (1.0 + bt)))
    # Thin-ring parallel box length Lz = pi lambda = 2 pi / (vA/U).
    assert lba.thin_ring_Lz(config) == pytest.approx(2.0 * np.pi / config.vA_over_U)


def test_invalid_physics_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="gamma_ad"):
        _config(gamma_ad=1.0)
    with pytest.raises(ValueError, match="vA_over_U"):
        _config(vA_over_U=-0.1)
    with pytest.raises(ValueError, match="cs2_over_vA2 > 0"):
        lba.derived_parameters(_config(cs2_over_vA2=0.0))


def test_input_file_parses_the_disc_physics_section(tmp_path: Path) -> None:
    input_file = tmp_path / "disc.input"
    input_file.write_text(
        """
[equations]
type = "low_beta_accretion"

[grid]
Nx = 8
Ny = 8
Nz = 8

[physics]
cs2_over_vA2 = 0.25
q_shear = 1.5
vA_over_U = 0.3
B_hat = -1.0
P_hat = -2.5
rho_hat = -1.5
gamma_ad = 1.4

[dissipation.drho]
nu_perp = 0.01
nu_par = 0.0
n_perp = 2
n_par = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.config.field_names == ["psi", "omega", "upar", "dbpar", "drho"]
    assert settings.initial_condition.type == "low_beta_accretion_mode"
    assert settings.config.vA_over_U == 0.3
    assert settings.config.q_shear == 1.5
    assert settings.config.B_hat == -1.0
    assert settings.config.P_hat == -2.5
    assert settings.config.rho_hat == -1.5
    assert settings.config.gamma_ad == 1.4
    assert settings.config.dissipation["drho"]["nu_perp"] == 0.01


# --------------------------------------------------------------------------
# linear behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k_indices", [(1, 2, 1), (0, 1, 1), (2, 0, 3), (1, 1, 2), (3, 2, 1)])
def test_ideal_rhs_reproduces_linear_matrix_on_a_single_mode(k_indices) -> None:
    """A single stored Fourier mode has all perpendicular gradients parallel to
    one `k_perp`, so every Poisson bracket vanishes and the full nonlinear RHS
    must equal `linear_matrix @ f` exactly."""

    config = _config()
    backend, grid, fft, workspace, mask = _context(config)
    ix, iy, iz = k_indices
    rng = np.random.default_rng(11)
    vector = rng.normal(size=5) + 1j * rng.normal(size=5)

    state = State(grid, backend, field_names=lba.FIELD_NAMES)
    for component, name in enumerate(lba.FIELD_NAMES):
        state[name][ix, iy, iz] = vector[component]

    rhs = lba.ideal_rhs(state, grid, fft, workspace, config, dealias_mask=mask)
    computed = np.array([backend.to_numpy(rhs[name])[ix, iy, iz] for name in lba.FIELD_NAMES])

    kx = backend.scalar_to_float(grid.kx[ix, 0, 0])
    ky = backend.scalar_to_float(grid.ky[0, iy, 0])
    kz = backend.scalar_to_float(grid.kz[0, 0, iz])
    expected = lba.linear_matrix(kx, ky, kz, config) @ vector

    np.testing.assert_allclose(computed, expected, rtol=1.0e-12, atol=1.0e-13)

    # ... and nothing is generated at any other wavenumber.
    for name in lba.FIELD_NAMES:
        residual = backend.to_numpy(rhs[name]).copy()
        residual[ix, iy, iz] = 0.0
        assert np.max(np.abs(residual)) < 1.0e-12 * max(1.0, np.max(np.abs(expected)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linear_matrix_matches_kawazura_dispersion_relation_at_zero_curvature(seed: int) -> None:
    """For `vA_over_U = 0` the eigenvalues must be the four roots of Kawazura
    et al. (2022) equation (3.2) plus one exactly marginal entropy mode."""

    rng = np.random.default_rng(seed)
    for _ in range(40):
        kx, ky, kz = rng.normal(size=3) * 2.0
        beta_tilde = float(10.0 ** rng.uniform(-1.5, 1.5))
        q = float(rng.uniform(0.0, 2.0))
        config = _config(
            cs2_over_vA2=beta_tilde,
            q_shear=q,
            vA_over_U=0.0,
            B_hat=float(rng.normal()),
            P_hat=float(rng.normal()),
            rho_hat=float(rng.normal()),
        )
        eigenvalues = np.linalg.eigvals(lba.linear_matrix(kx, ky, kz, config))

        lam = 1.0 + 1.0 / beta_tilde
        ratio = 2.0 * ky**2 / (kx**2 + ky**2)
        # (omega^2 - kz^2)(lam omega^2 - kz^2) = ratio [(2-q) lam omega^2 + q kz^2]
        omega = np.roots(
            [lam, 0.0, -((1.0 + lam) * kz**2 + ratio * (2.0 - q) * lam), 0.0, kz**4 - ratio * q * kz**2]
        )
        predicted = np.append(-1j * omega, 0.0)  # f ~ exp(lambda t) = exp(-i omega t)

        distance = np.abs(eigenvalues[:, None] - predicted[None, :])
        scale = 1.0 + np.abs(predicted).max()
        assert distance.min(axis=1).max() / scale < 1.0e-10
        assert distance.min(axis=0).max() / scale < 1.0e-10


@pytest.mark.parametrize("beta", [0.1, 1.0, 10.0, 100.0])
def test_maximum_mri_growth_rate_matches_kawazura_formula(beta: float) -> None:
    gamma = 5.0 / 3.0
    config = _config(
        cs2_over_vA2=gamma * beta / 2.0,
        gamma_ad=gamma,
        q_shear=1.5,
        vA_over_U=0.0,
        B_hat=0.0,
        P_hat=0.0,
        rho_hat=0.0,
    )
    # kx = 0 is the fastest-growing choice and the growth rate is then
    # independent of ky, so scan kz only.
    growth = max(
        float(np.linalg.eigvals(lba.linear_matrix(0.0, 1.0, kz, config)).real.max())
        for kz in np.linspace(1.0e-4, 3.0, 3000)
    )
    expected = np.sqrt(
        5.0 * beta / 18.0 * (20.0 * beta + 15.0 - np.sqrt(8.0 * (50.0 * beta**2 + 75.0 * beta + 18.0)))
    )
    assert growth == pytest.approx(expected, rel=1.0e-5)


@pytest.mark.parametrize(
    "mode,k_indices",
    [
        ("fastest_growing", (0, 1, 1)),
        ("highest_frequency", (1, 2, 1)),
        ("lowest_frequency", (1, 2, 1)),
        ("fastest_decaying", (0, 1, 1)),
    ],
)
def test_single_eigenmode_follows_exp_eigenvalue_t(mode: str, k_indices) -> None:
    config = _config(Lz=4.0 * np.pi)
    backend, grid, fft, workspace, mask = _context(config)
    state0 = low_beta_accretion_mode_state(
        grid=grid,
        backend=backend,
        field_names=config.field_names,
        k_indices=k_indices,
        amplitude=0.1,
        mode=mode,
        params=config,
    )
    eigenvalue = low_beta_accretion_mode_eigenvalue(grid, backend, k_indices, mode, config)

    dt, steps = 1.0e-3, 40
    evolved = _advance(
        state0, steps=steps, dt=dt, config=config, grid=grid, fft=fft, workspace=workspace, mask=mask
    )
    factor = np.exp(eigenvalue * steps * dt)

    for name in state0.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(evolved[name]),
            backend.to_numpy(state0[name]) * factor,
            rtol=1.0e-7,
            atol=1.0e-10,
            err_msg=f"Eigenmode branch {mode!r} mismatch in field {name}.",
        )


def test_eigenmode_initial_condition_is_normalized_to_the_requested_energy() -> None:
    config = _config()
    backend, grid, fft, workspace, mask = _context(config)
    state = build_initial_state(
        "low_beta_accretion_mode",
        parameters={"k_indices": [0, 1, 1], "amplitude": 0.25, "mode": "fastest_growing"},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    assert lba.total_energy(state, grid, backend, config) == pytest.approx(0.25**2, rel=1.0e-12)


def _linearized_energy_history(
    state: State, *, dt: float, n_steps: int, sample_every: int, config, grid, fft, workspace, mask
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve with the brackets switched off and sample `total_energy`.

    The run driver does the same substitution for `equation_mode = "linear"`;
    doing it here keeps the test independent of the driver.
    """

    from rmhdgpu.run import _zero_poisson_bracket

    backend = state.backend
    original_bracket = lba.poisson_bracket
    lba.poisson_bracket = _zero_poisson_bracket
    try:
        times, energies = [], []
        current = state
        for step in range(n_steps + 1):
            if step % sample_every == 0:
                times.append(step * dt)
                energies.append(lba.total_energy(current, grid, backend, config))
            current = _advance(
                current, steps=1, dt=dt, config=config, grid=grid, fft=fft, workspace=workspace, mask=mask
            )
    finally:
        lba.poisson_bracket = original_bracket
    return np.asarray(times), np.asarray(energies)


def _max_growth_rate(grid, backend, config, indices) -> float:
    return max(
        float(np.linalg.eigvals(lba.linear_matrix(
            backend.scalar_to_float(grid.kx[int(ix), 0, 0]),
            backend.scalar_to_float(grid.ky[0, int(iy), 0]),
            backend.scalar_to_float(grid.kz[0, 0, int(iz)]),
            config,
        )).real.max())
        for ix, iy, iz in indices
    )


@pytest.mark.parametrize("k_indices", [(0, 1, 2), (1, 2, 1)])
def test_random_single_mode_grows_at_that_modes_dispersion_rate(k_indices) -> None:
    """Random small-amplitude data in one Fourier mode must grow at the largest
    `Re(lambda)` of that mode's 5x5 dispersion matrix."""

    config = _config(Nx=8, Ny=8, Nz=8, Lz=4.0 * np.pi, vA_over_U=0.25)
    backend, grid, fft, workspace, mask = _context(config)
    ix, iy, iz = k_indices

    rng = np.random.default_rng(17)
    state = State(grid, backend, field_names=config.field_names)
    for name in config.field_names:
        state[name][ix, iy, iz] = 1.0e-6 * (rng.normal() + 1j * rng.normal())

    expected = _max_growth_rate(grid, backend, config, [k_indices])
    assert expected > 0.0

    times, energies = _linearized_energy_history(
        state,
        dt=5.0e-3,
        n_steps=6000,
        sample_every=200,
        config=config,
        grid=grid,
        fft=fft,
        workspace=workspace,
        mask=mask,
    )
    # Fit only after the subdominant branches have decayed away.
    window = times >= 20.0
    measured = 0.5 * np.polyfit(times[window], np.log(energies[window]), 1)[0]
    assert measured == pytest.approx(expected, rel=1.0e-5)


def test_random_field_growth_rate_converges_to_the_fastest_linear_mode() -> None:
    """A small band-limited random field must asymptote to the fastest growth
    rate among the modes it actually populates.

    Linear evolution keeps Fourier modes independent, so only populated modes
    can contribute; convergence is slow here because neighbouring modes have
    similar growth rates, hence the long integration and 1% tolerance.
    """

    config = _config(Nx=8, Ny=8, Nz=8, Lz=4.0 * np.pi, vA_over_U=0.25)
    backend, grid, fft, workspace, mask = _context(config)
    state = _random_state(config, grid, backend, fft, mask, seed=3, energy=1.0e-20)

    populated = np.zeros(grid.fourier_shape, dtype=bool)
    for name in state.field_names:
        populated |= np.abs(backend.to_numpy(state[name])) > 0.0
    expected = _max_growth_rate(grid, backend, config, zip(*np.nonzero(populated)))
    assert expected > 0.0

    times, energies = _linearized_energy_history(
        state,
        dt=5.0e-3,
        n_steps=10000,
        sample_every=250,
        config=config,
        grid=grid,
        fft=fft,
        workspace=workspace,
        mask=mask,
    )
    window = times >= 35.0
    measured = 0.5 * np.polyfit(times[window], np.log(energies[window]), 1)[0]
    assert measured == pytest.approx(expected, rel=3.0e-3)


# --------------------------------------------------------------------------
# energy budget
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"vA_over_U": 0.0},
        {"vA_over_U": 0.0, "q_shear": 0.0},
        {"cs2_over_vA2": 5.0, "gamma_ad": 1.4, "vA_over_U": 0.8, "B_hat": 2.0, "P_hat": 1.0, "rho_hat": 0.5},
    ],
)
def test_energy_source_terms_equal_the_exact_energy_rate(overrides) -> None:
    config = _config(Nx=16, Ny=16, Nz=16, **overrides)
    backend, grid, fft, workspace, mask = _context(config)
    state = _random_state(config, grid, backend, fft, mask, seed=7)
    rhs = lba.ideal_rhs(state, grid, fft, workspace, config, dealias_mask=mask)

    exact = _exact_energy_rate(state, rhs, grid, backend, config)
    terms = lba.total_energy_source_rhs_terms(state, grid, backend, config)

    assert set(terms) == set(lba.ENERGY_SOURCE_TERM_NAMES)
    assert sum(terms.values()) == pytest.approx(exact, rel=1.0e-10, abs=1.0e-14)


def test_only_the_shear_source_survives_the_kawazura_limit() -> None:
    config = _config(Nx=16, Ny=16, Nz=16, vA_over_U=0.0)
    backend, grid, fft, workspace, mask = _context(config)
    state = _random_state(config, grid, backend, fft, mask, seed=5)
    terms = lba.total_energy_source_rhs_terms(state, grid, backend, config)

    assert abs(terms["shear"]) > 1.0e-6
    for name in ("curvature", "buoyancy", "pressure_gradient", "entropy_gradient"):
        assert terms[name] == 0.0


def test_energy_is_exactly_conserved_in_the_homogeneous_limit() -> None:
    """With `vA_over_U = 0` and `q = 0` the system is homogeneous RMHD, so the
    nonlinear ideal evolution must conserve the free energy."""

    config = _config(Nx=16, Ny=16, Nz=16, vA_over_U=0.0, q_shear=0.0)
    backend, grid, fft, workspace, mask = _context(config)
    state = _random_state(config, grid, backend, fft, mask, seed=4, energy=1.0)

    initial = lba.total_energy(state, grid, backend, config)
    evolved = _advance(
        state, steps=200, dt=2.0e-3, config=config, grid=grid, fft=fft, workspace=workspace, mask=mask
    )
    final = lba.total_energy(evolved, grid, backend, config)
    assert abs(final - initial) / initial < 2.0e-8


def test_entropy_variable_is_advected_when_the_background_is_isentropic() -> None:
    """`sigma = dbpar/beta_tilde + drho` obeys `d sigma/dt = -mu Shat/gamma d_y phi`,
    so `<sigma^2>` is conserved when `Shat = Phat - gamma rhohat = 0`."""

    gamma = 5.0 / 3.0
    config = _config(Nx=16, Ny=16, Nz=16, gamma_ad=gamma, P_hat=gamma * (-1.3), rho_hat=-1.3)
    assert lba.derived_parameters(config).S_hat == pytest.approx(0.0)

    backend, grid, fft, workspace, mask = _context(config)
    state = _random_state(config, grid, backend, fft, mask, seed=9)

    def sigma_variance(s: State) -> float:
        sigma = lba.derive_entropy_hat(s, config)
        return modal_average(0.5 * backend.xp.abs(sigma) ** 2, grid, backend)

    initial = sigma_variance(state)
    evolved = _advance(
        state, steps=200, dt=2.0e-3, config=config, grid=grid, fft=fft, workspace=workspace, mask=mask
    )
    assert abs(sigma_variance(evolved) - initial) / initial < 2.0e-8


def test_dissipation_budget_term_matches_the_exact_energy_rate() -> None:
    """Equal damping on `dbpar` and `drho` keeps the compressive dissipation
    block negative definite; the reported term must equal the exact rate."""

    config = _config(
        Nx=16,
        Ny=16,
        Nz=16,
        dissipation={
            name: {"nu_perp": 0.02, "nu_par": 0.01, "n_perp": 1, "n_par": 1}
            for name in ["psi", "omega", "upar", "dbpar", "drho"]
        },
    )
    backend, grid, fft, workspace, mask = _context(config)
    state = _random_state(config, grid, backend, fft, mask, seed=13)
    linear_ops = lba.build_dissipation_operators(grid, config)

    damping_rhs = state.zeros_like()
    for name in state.field_names:
        damping_rhs[name][...] = -linear_ops[name] * state[name]

    exact = _exact_energy_rate(state, damping_rhs, grid, backend, config)
    reported = lba.total_energy_dissipation_rhs(state, grid, backend, linear_ops, config)
    assert reported == pytest.approx(exact, rel=1.0e-10)
    assert reported < 0.0


def test_scalar_diagnostics_expose_all_documented_names() -> None:
    config = _config()
    backend, grid, fft, workspace, mask = _context(config)
    state = _random_state(config, grid, backend, fft, mask, seed=2)
    linear_ops = lba.build_dissipation_operators(grid, config)

    diagnostics = lba.compute_equation_scalar_diagnostics(
        state,
        grid=grid,
        fft=fft,
        backend=backend,
        params=config,
        workspace=workspace,
        linear_ops=linear_ops,
    )

    expected = (
        "total_energy",
        "total_energy_rhs_total",
        "total_energy_rhs_dissipation",
        "total_energy_rhs_forcing",
        "alfvenic_energy",
        "compressive_energy",
        "entropy_energy",
        *(f"total_energy_rhs_{name}" for name in lba.ENERGY_SOURCE_TERM_NAMES),
    )
    for name in expected:
        assert name in diagnostics, name
        assert name in lba.SCALAR_DIAGNOSTIC_INFO, name

    # The three partitions add up to the total free energy.
    assert (
        diagnostics["alfvenic_energy"]
        + diagnostics["compressive_energy"]
        + diagnostics["entropy_energy"]
    ) == pytest.approx(diagnostics["total_energy"], rel=1.0e-12)


# --------------------------------------------------------------------------
# end-to-end run
# --------------------------------------------------------------------------


def _write_input(path: Path, *, tmax: float, dt: float, t_out_scal: float, extra: str = "") -> None:
    path.write_text(
        f"""
title = "Low-beta accretion test"
output_dir = "outputs"

[equations]
type = "low_beta_accretion"

[grid]
Nx = 8
Ny = 8
Nz = 8
Lz = 12.566370614359172

[time]
tmax = {tmax}
dt_init = {dt}
dt_max = {dt}
use_variable_dt = false

[output]
t_out_scal = {t_out_scal}

[backend]
backend = "numpy"

[runtime]
progress_output_every = 1000

[physics]
cs2_over_vA2 = 0.6
q_shear = 1.5
vA_over_U = 0.4
B_hat = -0.9
P_hat = -2.7
rho_hat = -1.3
gamma_ad = 1.6666666666666667

[initial_condition]
type = "low_beta_accretion_mode"

[initial_condition.parameters]
k_indices = [0, 1, 1]
mode = "fastest_growing"
amplitude = 0.02
{extra}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
        assert reader.fieldnames is not None
        return list(reader.fieldnames), rows


def test_run_writes_all_budget_columns_and_matches_finite_difference(tmp_path: Path) -> None:
    input_file = tmp_path / "disc_budget.input"
    _write_input(input_file, tmax=0.05, dt=0.001, t_out_scal=0.005)

    main([str(input_file)])

    fieldnames, rows = _read_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    for name in lba.ENERGY_SOURCE_TERM_NAMES:
        assert f"total_energy_rhs_{name}" in fieldnames

    time = np.asarray([row["time"] for row in rows])
    energy = np.asarray([row["total_energy"] for row in rows])
    rhs_total = np.asarray([row["total_energy_rhs_total"] for row in rows])

    measured = np.diff(energy) / np.diff(time)
    np.testing.assert_allclose(measured, rhs_total[1:], rtol=0.02, atol=1.0e-12)
    # The eigenmode is growing, so the budget is genuinely nonzero.
    assert np.all(rhs_total[1:] > 0.0)
