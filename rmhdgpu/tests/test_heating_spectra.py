"""Heating bins must close the physical energy budget with runtime operators."""
import csv

import numpy as np
import pytest

from rmhdgpu.auto_dissipation import AutoDissipationController
from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics.spectra import perpendicular_energy_spectrum_from_state
from rmhdgpu.equations import get_equation_module
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.testing import random_spectrum_test_parameters
from rmhdgpu.run import main
from rmhdgpu.runfile import dump_toml
from rmhdgpu.state import State


EQUATIONS = ("alfvenic", "s09", "low_beta_stratified")


def context(equation, backend_name, precision="float64", nz=10):
    eq = get_equation_module(equation)
    manual = {name: dict(nu_perp=.003*(i+1), nu_par=.007*(i+1),
                        n_perp=1+i % 3, n_par=1+i % 2)
              for i, name in enumerate(eq.FIELD_NAMES)}
    config = Config(equation_set=equation, Nx=8, Ny=10, Nz=nz,
        Lx=2*np.pi, Ly=4*np.pi, Lz=3*np.pi, backend=backend_name,
        fft_workers=2, real_dtype=precision,
        complex_dtype="complex64" if precision == "float32" else "complex128",
        cs2_over_vA2=.3, N2=.4, dissipation=manual,
        auto_dissipation=dict(mode="auto", n_perp=3, n_par=2, nu_par=.004,
            smooth_factor=1., max_update_factor=1., update_every=1))
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    state = State(grid, backend, field_names=config.field_names)
    return config, eq, backend, grid, fft, state


def spectra(context, operators, *, bin_width=.8):
    config, eq, backend, grid, _, state = context
    return perpendicular_energy_spectrum_from_state(
        state, grid, backend, equation_module=eq, params=config,
        bin_width=bin_width, linear_ops=operators)


@pytest.mark.parametrize("equation", EQUATIONS)
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("nz", [8, 10])
@pytest.mark.parametrize("mode", ["manual", "auto"])
def test_heating_sum_matches_instantaneous_budget(equation, precision, nz, mode, production_backend):
    ctx = context(equation, production_backend, precision, nz)
    config, eq, backend, grid, fft, state = ctx
    # Real FFTs exercise conjugate weights, kz=0 and even-Nz Nyquist planes.
    # Non-square boxes and multiple modes per bin rule out bin-center damping.
    rng = np.random.default_rng(23)
    for name in state.field_names:
        real = backend.asarray(.03*rng.normal(size=grid.real_shape), dtype=grid.real_dtype)
        state[name][...] = fft.r2c(real)
        if name == "omega":
            state[name][...] *= -grid.kperp2  # the generated real field was phi

    controller = AutoDissipationController.from_runtime(
        settings=config.auto_dissipation, equation_module=eq,
        field_names=config.field_names, grid=grid, backend=backend, dealias_mask=None)
    rtol = 3e-6 if precision == "float32" else 3e-14
    coefficients = []
    for amplitude in (1., 1.7):
        for name in state.field_names:
            state[name][...] *= amplitude
        if mode == "auto":
            controller.update(state, config)
            coefficients.append(controller.current_nu_perp)
            operators = eq.build_dissipation_operators(
                grid, config, dissipation_spec=controller.effective_dissipation())
        else:
            operators = eq.build_dissipation_operators(grid, config)
        result = spectra(ctx, operators)
        heat = result["heating_total"]
        expected = -eq.total_energy_dissipation_rhs(state, grid, backend, operators, config)
        assert expected > 0
        assert np.all(heat >= 0)
        np.testing.assert_allclose(np.sum(heat), expected, rtol=rtol)
        # A deliberately different interval mean must not replace the new scalar.
        scalar = eq.compute_equation_scalar_diagnostics(state, grid=grid, fft=fft,
            backend=backend, params=config, linear_ops=operators,
            extra_rhs_terms={"total_energy": {"dissipation": -222.}},
            budget_rhs_terms={"total_energy": {"dissipation": -123.}})
        assert scalar["total_energy_rhs_dissipation"] == -123.
        np.testing.assert_allclose(scalar["heating_total"], heat.sum(), rtol=rtol)
        # Adding heating leaves all existing energy spectra unchanged.
        energy_only = spectra(ctx, None)
        assert "heating_total" not in energy_only
        for name in energy_only:
            np.testing.assert_array_equal(result[name], energy_only[name])
        if mode == "auto":
            stale = spectra(ctx, eq.build_dissipation_operators(grid, config))
            assert not np.isclose(heat.sum(), stale["heating_total"].sum(), rtol=.01)
    if mode == "auto":
        assert coefficients[1] > coefficients[0]


COMPONENTS = [(eq, field) for eq in EQUATIONS
              for field in get_equation_module(eq).FIELD_NAMES]


@pytest.mark.parametrize("equation,field", COMPONENTS)
def test_analytic_mode_heating_and_bin_location(equation, field, production_backend):
    ctx = context(equation, production_backend)
    config, eq, backend, grid, fft, state = ctx
    # Two distinct perpendicular and parallel wavenumbers occupy the same bin.
    # Their analytic loss checks field weights and the amplitude-to-energy 2.
    modes = [(1, 0, 1, .2), (1, 1, 2, .3)]
    x, y, z = np.indices(grid.real_shape)
    real = sum(a*np.cos(2*np.pi*(nx*x/grid.Nx + ny*y/grid.Ny + nz*z/grid.Nz))
               for nx, ny, nz, a in modes)
    state[field][...] = fft.r2c(backend.asarray(real, dtype=grid.real_dtype))
    if field == "omega":
        state[field][...] *= -grid.kperp2
    spec = config.dissipation[field]
    expected = 0.
    for nx, ny, nz, amplitude in modes:
        kp2 = (2*np.pi*nx/grid.Lx)**2 + (2*np.pi*ny/grid.Ly)**2
        kz2 = (2*np.pi*nz/grid.Lz)**2
        weight = {"psi": kp2, "omega": kp2, "upar": 1.,
                  "dbpar": (1+config.cs2_over_vA2)/config.cs2_over_vA2,
                  "s": config.cs2_over_vA2/((5/3)**2*(5/3-1)),
                  "a": 1/config.N2}[field]
        damping = spec["nu_perp"]*kp2**spec["n_perp"] + spec["nu_par"]*kz2**spec["n_par"]
        expected += damping*weight*amplitude**2/2
    result = spectra(ctx, eq.build_dissipation_operators(grid, config), bin_width=2.)
    np.testing.assert_allclose(result["heating_total"][0], expected, rtol=3e-14)
    np.testing.assert_allclose(result["heating_total"][1:], 0., atol=1e-27)


@pytest.mark.parametrize("equation", EQUATIONS)
def test_zero_damping_gives_zero_heating(equation):
    ctx = context(equation, "numpy")
    config, eq, backend, grid, fft, state = ctx
    for name in state.field_names:
        state[name][...] = 1. + 2j
    operators = {name: backend.xp.zeros_like(grid.kperp2) for name in state.field_names}
    np.testing.assert_array_equal(spectra(ctx, operators)["heating_total"], 0.)


def test_signed_low_beta_energy_is_not_clipped():
    ctx = context("low_beta_stratified", "numpy")
    config, eq, backend, grid, fft, state = ctx
    config.N2 = -.4
    state["a"][0, 0, 1] = 10.
    operators = eq.build_dissipation_operators(grid, config)
    heat = spectra(ctx, operators)["heating_total"]
    assert heat[0] < 0  # parallel damping of a negative-energy k_perp=0 mode
    np.testing.assert_allclose(heat.sum(),
        -eq.total_energy_dissipation_rhs(state, grid, backend, operators, config), rtol=3e-14)


@pytest.mark.parametrize("equation", EQUATIONS)
@pytest.mark.parametrize("mode", ["manual", "auto"])
def test_saved_spectra_sum_to_saved_instantaneous_heating(tmp_path, equation, mode, production_backend):
    eq = get_equation_module(equation)
    dissipation = {name: dict(nu_perp=.001*(i+1), nu_par=.002*(i+1), n_perp=2, n_par=1)
                   for i, name in enumerate(eq.FIELD_NAMES)}
    dissipation.update(mode=mode, n_perp=3, n_par=1, nu_par=.004,
                       smooth_factor=1., max_update_factor=1., update_every=1)
    document = dict(equations=dict(type=equation), grid=dict(Nx=8, Ny=8, Nz=8),
        backend=dict(backend=production_backend, fft_workers=2),
        physics=dict(cs2_over_vA2=.3, N2=.4),
        time=dict(tmax=.004, dt_init=.001, use_variable_dt=False),
        output=dict(t_out_scal=.002, t_out_spec=.001, t_out_full=0.),
        runtime=dict(progress_output_every=100), dissipation=dissipation,
        initial_condition=dict(type="random_spectrum", parameters=random_spectrum_test_parameters(.03)))
    path = tmp_path / "heating.input"
    path.write_text(dump_toml(document))
    main([str(path)])
    with (tmp_path / "outputs" / "scalar_diagnostics.csv").open() as f:
        scalars = list(csv.DictReader(f))
    with (tmp_path / "outputs" / "spectra.csv").open() as f:
        rows = list(csv.DictReader(f))
    totals = {}
    for row in rows:
        if row["quantity"] == "heating_total":
            identity = (float(row["time"]), int(row["step"]))
            totals[identity] = totals.get(identity, 0.) + float(row["value"])
    assert len(totals) == 5  # spectra also work between scalar outputs
    for row in scalars:
        identity = (float(row["time"]), int(row["step"]))
        np.testing.assert_allclose(totals[identity], float(row["heating_total"]), rtol=3e-14)
    assert float(scalars[0]["heating_total"]) > 0
    assert float(scalars[0]["total_energy_rhs_dissipation"]) == 0  # no preceding interval
    if mode == "auto":
        assert float(scalars[-1]["auto_dissipation_nu_perp"]) != float(scalars[0]["auto_dissipation_nu_perp"])
