"""Qualitative sanity check for nonlinear exact Alfvén-wave propagation.

Run with:

`python -m rmhdgpu.examples.sanity_aw_packet`

The script builds a large-amplitude `phi = psi` Alfvénic packet from the first
three Fourier modes in each direction, advances it ideally, and overplots a
fixed `(x, y)` slice at times spaced by `0.1` Alfvén times.
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
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import lap_perp
from rmhdgpu.state import State
from rmhdgpu.steppers import if_ssprk3_step
from rmhdgpu.workspace import Workspace


def _packet_real_field(grid: object, backend: object) -> object:
    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)
    field = xp.zeros(grid.real_shape, dtype=grid.real_dtype)

    for nx in range(1, 4):
        for ny in range(1, 4):
            for nz in range(1, 4):
                coefficient = 0.15 / (nx + ny + nz - 1.0)
                phase = 0.3 * (nx - ny + nz)
                field += coefficient * xp.cos(nx * x + ny * y + nz * z + phase)

    rms = backend.scalar_to_float(xp.sqrt(xp.mean(field**2)))
    field *= 1.0 / rms
    return field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sanity_plots", help="Directory where figures are written.")
    parser.add_argument("--nx", type=int, default=32, help="Grid resolution in each direction.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(Nx=args.nx, Ny=args.nx, Nz=args.nx, backend="numpy", use_variable_dt=False, dt_init=0.07e-2)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    linear_ops = s09.build_dissipation_operators(grid, config)

    phi_real = _packet_real_field(grid, backend)
    phi_hat = fft.r2c(phi_real) * mask

    state = State(grid, backend, field_names=config.field_names)
    state["psi"][...] = phi_hat
    state["omega"][...] = lap_perp(phi_hat, grid)

    tau_A = grid.Lz / config.vA
    sample_times = np.arange(0.0, 0.51 * tau_A, 0.1 * tau_A)
    samples: list[tuple[float, np.ndarray]] = []
    slice_x = grid.Nx // 3
    slice_y = grid.Ny // 4

    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    t = 0.0
    sample_index = 0
    while sample_index < len(sample_times):
        if t >= sample_times[sample_index] - 1.0e-15:
            psi_real = backend.to_numpy(fft.c2r(state["psi"]))
            samples.append((sample_times[sample_index] / tau_A, psi_real[slice_x, slice_y, :].copy()))
            sample_index += 1
            continue

        dt = min(config.dt_init, sample_times[sample_index] - t)
        state = if_ssprk3_step(state, dt, s09.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)
        t += dt

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(samples)))
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    z = backend.to_numpy(grid.z)
    for color, (time_tau_A, profile) in zip(colors, samples, strict=True):
        ax.plot(z, profile, color=color, lw=2, label=f"t/tA={time_tau_A:.1f}")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\psi(x_0, y_0, z)$")
    ax.set_title("Exact nonlinear Alfvén-wave packet translation")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    figure_path = output_dir / "sanity_aw_packet.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
