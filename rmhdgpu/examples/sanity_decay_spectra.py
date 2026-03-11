"""Qualitative decaying-turbulence spectra sanity check.

Run with:

`python -m rmhdgpu.examples.sanity_decay_spectra`

The script performs a modest dissipative run from low-mode initial data and
saves perpendicular shell spectra at several times.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics.spectra import perpendicular_energy_spectrum_from_state
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import dx, dy, lap_perp
from rmhdgpu.state import State
from rmhdgpu.steppers import compute_cfl_timestep, if_ssprk3_step
from rmhdgpu.workspace import Workspace


def estimate_hyperdiffusion_coefficient(k_d: float, k0: float, u_rms: float, order: int) -> float:
    """Estimate `nu` from `tau_nl^{-1}(k_d) ~ nu * k_d^(2 n)`.

    The nonlinear rate estimate used is

    `tau_nl^{-1}(k_d) ~ k_d * u_rms * (k_d / k0)^(-1/3)`

    so the coefficient is chosen as

    `nu ~ k_d * u_rms * (k_d / k0)^(-1/3) / k_d^(2 n)`.
    """

    return k_d * u_rms * (k_d / k0) ** (-1.0 / 3.0) / (k_d ** (2 * order))


def _low_mode_real_field(
    grid: object,
    backend: object,
    seed: int,
    amplitude: float,
) -> object:
    rng = np.random.default_rng(seed)
    x = backend.to_numpy(grid.x).reshape(grid.Nx, 1, 1)
    y = backend.to_numpy(grid.y).reshape(1, grid.Ny, 1)
    z = backend.to_numpy(grid.z).reshape(1, 1, grid.Nz)
    field = np.zeros(grid.real_shape, dtype=np.float64)

    for nx in range(1, 4):
        for ny in range(1, 4):
            for nz in range(1, 4):
                a_cos = rng.normal(scale=amplitude / 6.0)
                a_sin = rng.normal(scale=amplitude / 6.0)
                phase = nx * x + ny * y + nz * z
                field += a_cos * np.cos(phase) + a_sin * np.sin(phase)

    return backend.asarray(field.astype(grid.real_dtype, copy=False))


def _initial_u_rms(phi_hat: object, grid: object, fft: FFTManager, backend: object) -> float:
    ux = -fft.c2r(dy(phi_hat, grid))
    uy = fft.c2r(dx(phi_hat, grid))
    xp = backend.xp
    return backend.scalar_to_float(xp.sqrt(xp.mean(ux**2 + uy**2)))


def dealiased_max_kperp(grid: object, backend: object, mask: object) -> float:
    """Return the maximum retained perpendicular wavenumber after 2/3 dealiasing."""

    kperp = np.sqrt(backend.to_numpy(grid.kperp2))
    retained = backend.to_numpy(mask).astype(bool)
    return float(np.max(kperp[retained]))


def _plot_spectra(
    spectra_by_time: list[tuple[float, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    keys = ["u_perp", "b_perp", "upar", "dbpar", "s"]
    titles = {
        "u_perp": r"$E_{u_\perp}(k_\perp)$",
        "b_perp": r"$E_{b_\perp}(k_\perp)$",
        "upar": r"$E_{u_\parallel}(k_\perp)$",
        "dbpar": r"$E_{\delta B_\parallel}(k_\perp)$",
        "s": r"$E_s(k_\perp)$",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes_flat = list(axes.flat)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(spectra_by_time)))

    for axis, key in zip(axes_flat[: len(keys)], keys, strict=True):
        for color, (time, spectra) in zip(colors, spectra_by_time, strict=True):
            mask = spectra["kperp"] > 0.0
            axis.loglog(spectra["kperp"][mask], spectra[key][mask], color=color, lw=2, label=f"t={time:.2f}")

        reference_k = spectra_by_time[0][1]["kperp"]
        mask = reference_k > 1.0
        if np.any(mask):
            k_ref = reference_k[mask]
            y_ref = 1.0e-3 * (k_ref / k_ref[0]) ** (-5.0 / 3.0)
            axis.loglog(k_ref, y_ref, "k--", alpha=0.5, label=r"$k^{-5/3}$")

        axis.set_title(titles[key])
        axis.set_xlabel(r"$k_\perp$")
        axis.grid(True, alpha=0.25)

    axes_flat[-1].axis("off")
    axes_flat[0].legend(fontsize=8)

    figure_path = output_dir / "sanity_decay_spectra.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sanity_plots", help="Directory where figures are written.")
    parser.add_argument("--n", type=int, default=96, help="Grid resolution in each direction.")
    parser.add_argument("--t-final", type=float, default=0.5, help="Final time.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        Nx=args.n,
        Ny=args.n,
        Nz=args.n,
        backend="numpy",
        cfl_number=0.5,
        dt_max=2.0e-2,
        tmax=args.t_final,
        use_variable_dt=True,
    )

    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)

    state = State(grid, backend, field_names=config.field_names)
    phi_hat = fft.r2c(_low_mode_real_field(grid, backend, seed=1, amplitude=0.4)) * mask
    psi_hat = fft.r2c(_low_mode_real_field(grid, backend, seed=2, amplitude=0.3)) * mask
    state["psi"][...] = psi_hat
    state["omega"][...] = lap_perp(phi_hat, grid)
    state["upar"][...] = fft.r2c(_low_mode_real_field(grid, backend, seed=3, amplitude=0.08)) * mask
    state["dbpar"][...] = fft.r2c(_low_mode_real_field(grid, backend, seed=4, amplitude=0.06)) * mask
    state["s"][...] = fft.r2c(_low_mode_real_field(grid, backend, seed=5, amplitude=0.05)) * mask

    u_rms = _initial_u_rms(phi_hat, grid, fft, backend)
    k0 = 1.0
    kperp_max_dealiased = dealiased_max_kperp(grid, backend, mask)
    # Choose the dissipation scale at roughly half the dealiased perpendicular
    # maximum so the damping acts near the top of the retained inertial range
    # rather than all the way at the truncation boundary.
    k_d = 0.5 * kperp_max_dealiased
    nu_perp = estimate_hyperdiffusion_coefficient(k_d=k_d, k0=k0, u_rms=u_rms, order=3)
    nu_par = 0.1 * nu_perp
    print(
        "dissipation-scale setup",
        {
            "k0": k0,
            "kperp_max_dealiased": kperp_max_dealiased,
            "k_d": k_d,
            "nu_perp": nu_perp,
            "nu_par": nu_par,
        },
    )
    for name in config.field_names:
        config.dissipation[name]["nu_perp"] = nu_perp
        config.dissipation[name]["nu_par"] = nu_par
        config.dissipation[name]["n_perp"] = 3
        config.dissipation[name]["n_par"] = 1

    linear_ops = s09.build_dissipation_operators(grid, config)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    sample_times = np.linspace(0.0, config.tmax, 6)
    spectra_by_time: list[tuple[float, dict[str, np.ndarray]]] = []

    current = state
    t = 0.0
    sample_index = 0
    while sample_index < len(sample_times):
        if t >= sample_times[sample_index] - 1.0e-15:
            spectra_by_time.append(
                (sample_times[sample_index], perpendicular_energy_spectrum_from_state(current, grid, backend))
            )
            sample_index += 1
            continue

        dt = compute_cfl_timestep(current, grid, fft, config)
        dt = min(dt, sample_times[sample_index] - t)
        current = if_ssprk3_step(current, dt, s09.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)
        t += dt

    _plot_spectra(spectra_by_time, output_dir)


if __name__ == "__main__":
    main()
