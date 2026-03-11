"""Basic spectral differential operators."""

from __future__ import annotations

from typing import Any

from rmhdgpu.masks import apply_mask


def dx(f_hat: Any, grid: Any) -> Any:
    """Return the Fourier-space x-derivative, `i k_x f_hat`."""

    return 1j * grid.kx * f_hat


def dy(f_hat: Any, grid: Any) -> Any:
    """Return the Fourier-space y-derivative, `i k_y f_hat`."""

    return 1j * grid.ky * f_hat


def dz(f_hat: Any, grid: Any) -> Any:
    """Return the Fourier-space z-derivative, `i k_z f_hat`."""

    return 1j * grid.kz * f_hat


def lap_perp(f_hat: Any, grid: Any) -> Any:
    """Return the perpendicular Laplacian in Fourier space.

    The convention is

    `lap_perp(f_hat) = -k_perp^2 * f_hat`

    so that the inverse operator below satisfies

    `inv_lap_perp(lap_perp(f_hat)) = f_hat`

    on all modes with `k_perp != 0`.
    """

    return -grid.kperp2 * f_hat


def inv_lap_perp(f_hat: Any, grid: Any) -> Any:
    """Return the inverse perpendicular Laplacian in Fourier space.

    This uses

    `inv_lap_perp(f_hat) = -inv_kperp2 * f_hat`

    with the `k_perp = 0` modes regularized to zero.
    """

    return -grid.inv_kperp2 * f_hat


def poisson_bracket(
    f_hat: Any,
    g_hat: Any,
    grid: Any,
    fft: Any,
    workspace: Any,
    mask: Any | None = None,
    out: Any | None = None,
) -> Any:
    """Compute the Fourier-space Poisson bracket `{f, g}`.

    The bracket is defined by

    `{f, g} = d_x f d_y g - d_y f d_x g`

    The result is written to `out` if provided. Otherwise the reusable
    workspace complex buffer `c0` is used and returned. That buffer will be
    overwritten by later calls.
    """

    dx_f_hat = dx(f_hat, grid)
    dy_f_hat = dy(f_hat, grid)
    dx_g_hat = dx(g_hat, grid)
    dy_g_hat = dy(g_hat, grid)

    r0 = workspace.real["r0"]
    r1 = workspace.real["r1"]
    r2 = workspace.real["r2"]
    r3 = workspace.real["r3"]
    r4 = workspace.real["r4"]

    fft.c2r(dx_f_hat, out=r0)
    fft.c2r(dy_f_hat, out=r1)
    fft.c2r(dx_g_hat, out=r2)
    fft.c2r(dy_g_hat, out=r3)

    r4[...] = r0 * r3 - r1 * r2

    if out is None:
        out = workspace.complex["c0"]
    fft.r2c(r4, out=out)

    if mask is not None:
        apply_mask(out, mask)

    return out

