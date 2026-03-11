from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.forcing import (
    forcing_shell_mask,
    generate_forcing_kick,
    mode_number_magnitude,
    shaped_random_real_field,
)
from rmhdgpu.grid import build_grid
from rmhdgpu.state import State


def _build_context() -> tuple[Config, object, object, FFTManager]:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="numpy",
        use_forcing=True,
        n_min_force=1.0,
        n_max_force=2.0,
        alpha_force=0.5,
        forcing_seed=1234,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    return config, backend, grid, fft


def test_forcing_shell_mask_selects_correct_band() -> None:
    _, backend, grid, _ = _build_context()
    mask = backend.to_numpy(forcing_shell_mask(grid, backend, 1.0, 2.0))
    n_mag = backend.to_numpy(mode_number_magnitude(grid, backend))

    assert mask.shape == grid.fourier_shape
    assert not mask[0, 0, 0]
    assert mask[1, 0, 0]
    assert mask[7, 0, 0]
    assert mask[1, 1, 0]
    assert not mask[0, 0, 3]
    assert not mask[2, 2, 0]
    assert np.all(mask[(n_mag >= 1.0) & (n_mag <= 2.0)])


def test_shaped_random_real_field_is_real_and_finite() -> None:
    config, backend, grid, fft = _build_context()
    real_field, field_hat = shaped_random_real_field(
        grid,
        backend,
        fft,
        n_min_force=config.n_min_force,
        n_max_force=config.n_max_force,
        alpha_force=config.alpha_force,
        rng=np.random.default_rng(11),
    )

    real_np = backend.to_numpy(real_field)
    recovered = backend.to_numpy(fft.c2r(field_hat))
    rms = np.sqrt(np.mean(real_np**2))

    assert np.all(np.isfinite(real_np))
    np.testing.assert_allclose(recovered, real_np, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(rms, 1.0, atol=1.0e-12, rtol=1.0e-12)


def test_shaped_random_real_field_respects_band() -> None:
    config, backend, grid, fft = _build_context()
    _, field_hat = shaped_random_real_field(
        grid,
        backend,
        fft,
        n_min_force=config.n_min_force,
        n_max_force=config.n_max_force,
        alpha_force=config.alpha_force,
        rng=np.random.default_rng(22),
    )

    mask = backend.to_numpy(forcing_shell_mask(grid, backend, config.n_min_force, config.n_max_force))
    field_hat_np = backend.to_numpy(field_hat)
    inside_power = float(np.sum(np.abs(field_hat_np[mask]) ** 2))
    outside_power = float(np.sum(np.abs(field_hat_np[~mask]) ** 2))

    assert inside_power > 0.0
    assert outside_power <= 1.0e-24, f"Power leaked outside forcing band: outside_power={outside_power:.3e}"


def test_force_amplitude_zero_gives_zero_kick() -> None:
    config, backend, grid, fft = _build_context()
    config.force_amplitudes = {name: 0.0 for name in config.field_names}
    state = State(grid, backend, field_names=config.field_names)

    kick = generate_forcing_kick(state, grid, fft, backend, config, np.random.default_rng(33), dt=0.1)

    for field_name in state.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(kick[field_name]),
            0.0,
            atol=0.0,
            rtol=0.0,
            err_msg=f"Expected zero forcing kick for {field_name} when sigma=0.",
        )


def test_forcing_kick_scales_like_sqrt_dt() -> None:
    config, backend, grid, fft = _build_context()
    config.force_amplitudes = {name: 0.0 for name in config.field_names}
    config.force_amplitudes["psi"] = 0.7
    state = State(grid, backend, field_names=config.field_names)
    dt1 = 0.05
    dt2 = 0.2

    kick1 = generate_forcing_kick(state, grid, fft, backend, config, np.random.default_rng(44), dt=dt1)
    kick2 = generate_forcing_kick(state, grid, fft, backend, config, np.random.default_rng(44), dt=dt2)

    rms1 = np.sqrt(np.mean(backend.to_numpy(fft.c2r(kick1["psi"])) ** 2))
    rms2 = np.sqrt(np.mean(backend.to_numpy(fft.c2r(kick2["psi"])) ** 2))

    np.testing.assert_allclose(rms2 / rms1, np.sqrt(dt2 / dt1), atol=1.0e-12, rtol=1.0e-12)


def test_forcing_reproducibility_with_seed() -> None:
    config, backend, grid, fft = _build_context()
    config.force_amplitudes["psi"] = 0.4
    config.force_amplitudes["upar"] = 0.2
    state = State(grid, backend, field_names=config.field_names)

    kick1 = generate_forcing_kick(state, grid, fft, backend, config, np.random.default_rng(55), dt=0.1)
    kick2 = generate_forcing_kick(state, grid, fft, backend, config, np.random.default_rng(55), dt=0.1)

    for field_name in state.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(kick1[field_name]),
            backend.to_numpy(kick2[field_name]),
            atol=1.0e-12,
            rtol=1.0e-12,
            err_msg=f"Forcing kick for {field_name} was not reproducible with a fixed seed.",
        )
