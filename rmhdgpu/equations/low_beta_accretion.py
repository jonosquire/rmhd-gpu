"""Low-beta accretion-disc RMHD equation set (rotating RMHD with radial gradients).

This is the local RMHD system for a strongly magnetised accretion disc whose
mean field is (nearly) azimuthal. It is obtained from the general-geometry
RMHD equations by dropping *parallel* background gradients, which is exact for
an axisymmetric disc with `B = B(r) bhat_phi` and `U = U(r) bhat_phi`. It
generalises the rotating RMHD (RRMHD) model of Kawazura et al. (2022, JPP 88,
905880311) by keeping terms of order `vA / U` -- field-line curvature and the
radial gradients of `B`, `p` and `rho` -- which reintroduces buoyancy and makes
the density perturbation a dynamical field.

Geometry and coordinates
------------------------
The local orthonormal triad is right handed, `xhat x yhat = bhat`, with

- `bhat = phihat`  (azimuthal; the code's parallel direction `z`)
- `xhat = rhat`    (radial; all background gradients point along `xhat`)
- `yhat = -Zhat`   (minus the rotation axis, i.e. "vertical")

so that `bhat . grad bhat = kappa = -xhat / r` and `bhat x kappa = -yhat / r`.
Because all background gradients are radial and the perpendicular velocity is
`u_perp = bhat x grad_perp phi` (i.e. `u_x = -d_y phi`, `u_y = d_x phi`), every
background-gradient source term below carries a `d_y`. This is the same
coordinate convention as Kawazura et al. (2022, figure 1).

Normalisation
-------------
All fields are normalised exactly as in the Kawazura et al. (2022) paper:

`t Omega -> t`, `x/L_perp -> x`, `y/L_perp -> y`, `z Omega / vA -> z`,
`Phi / (L_perp^2 Omega) -> Phi`, `Psi / (L_perp^2 Omega) -> Psi`,
`du_par / (L_perp Omega) -> upar`,
`vA dB_par / (B L_perp Omega) -> dbpar`,
`vA drho / (L_perp Omega rho) -> drho`.

After this rescaling `Omega = 1` and `vA` drops out of the equations entirely:
the config value `vA` is *not* used by this equation set, and the normalised
parallel Alfven speed is 1.

`vA_over_U` and the box length
------------------------------
A real ring closes on itself, `Lz = 2 pi r`, which in units of `vA/Omega` reads

    Lz = 2 pi r Omega / vA = 2 pi U / vA = 2 pi / (vA/U),

so for a whole-ring box `vA_over_U` and `Lz` would carry the same information
(`thin_ring_Lz(params)` returns that value). They are nevertheless kept as
*independent* inputs, and `vA_over_U` is the one to think of as the physical
parameter:

- A simulation box is normally a sub-arc of the ring, not the whole ring. Its
  modes are then simply shorter-wavelength modes of the same disc; each one
  depends on `k_par` and `vA/U`, not on `Lz`.
- Linearly the two choices are equivalent -- a mode does not know how big the
  box is -- but nonlinearly they are not: `Lz` fixes the largest parallel scale
  and hence which modes interact, so it is a numerical choice like `Lx` and
  `Ly`.
- This mirrors standard RMHD, where the perpendicular box size is degenerate
  with the fluctuation amplitude and so is held fixed while physical parameters
  are varied, rather than being retuned.

So: set `vA_over_U` to select the disc, and choose `Lz` for resolution. The
default `vA_over_U = 0` is the straight-field limit, i.e. the Kawazura et al.
(2022) system.

Evolved fields and equations
----------------------------
The evolved fields are `[psi, omega, upar, dbpar, drho]` with
`phi = inv_lap_perp(omega)`. Writing `d/dt = d_t + {phi, .}` for the total
derivative and `nabla_par = d_z + {psi, .}` for the total parallel derivative,
the ideal equations are

- `d/dt lap_perp phi = nabla_par lap_perp psi`
                      `- 2 d_y upar + 2 mu d_y dbpar - mu chi d_y drho`
- `d/dt psi          = d_z phi`
- `d/dt upar         = nabla_par dbpar - mu (Bhat + 1) d_y psi + (2 - q) d_y phi`
- `(1 + 1/bt) d/dt dbpar = nabla_par upar + q d_y psi + mu C_b d_y phi`
- `(1 + bt) d/dt drho    = -nabla_par upar - q d_y psi - mu C_rho d_y phi`

with

- `mu   = vA / U = 2 / lambda`      (`lambda` of the notes; a free parameter)
- `bt   = cs^2 / vA^2`              (config `cs2_over_vA2`; `beta = 2 bt / gamma`)
- `q    = -d ln Omega / d ln r`     (3/2 for Keplerian)
- `Bhat = d ln B / d ln r`, `Phat = d ln p / d ln r`, `rhohat = d ln rho / d ln r`
- `chi   = (U^2 - U_K^2) / vA^2 = bt Phat / gamma + Bhat + 1`  (radial force balance)
- `C_b   = Bhat - 1 - Phat / gamma`
- `C_rho = Bhat - 1 + bt Phat / gamma - (1 + bt) rhohat`

Setting `vA_over_U = 0` recovers Kawazura et al. (2022) equations (2.3a)-(2.3d)
exactly, with `drho` then decoupled; setting `vA_over_U = 0` and `q = 0`
recovers homogeneous RMHD, for which the energy below is exactly conserved.

Free energy
-----------
The quadratic free energy (normalised by `rho L_perp^2 Omega^2`) is positive
definite:

`E = <0.5 |grad_perp phi|^2 + 0.5 |grad_perp psi|^2 + 0.5 upar^2`
`    + 0.5 (1 + 1/bt) dbpar^2 + 0.5 bt/(gamma - 1) sigma^2>`

where `sigma = dbpar / bt + drho` is (minus 1/gamma times) the entropy
perturbation `ds/c_v`. The last two pieces are, respectively, the slow-mode
magnetic-plus-pressure energy `rho dVpar^2 / 2` and the entropy-mode energy
`p/(2 gamma (gamma - 1)) (ds/c_v)^2` of the general theory.

`E` is *not* conserved: background gradients exchange energy with the
fluctuations through five source terms, reported individually in the budget
diagnostics as `total_energy_rhs_<name>`:

- `shear`             `q (<dbpar d_y psi> - <upar d_y phi>)`      (Reynolds + Maxwell stress against `kappa + grad_perp ln U`)
- `curvature`         `mu (1 + Bhat) (<dbpar d_y phi> - <upar d_y psi>)`  (exchange with background magnetic energy)
- `buoyancy`          `-mu chi <drho d_y phi>`                    (work against the effective gravity)
- `pressure_gradient` `-mu Phat / gamma <dbpar d_y phi>`
- `entropy_gradient`  `-mu bt Shat / (gamma (gamma - 1)) <sigma d_y phi>`, `Shat = Phat - gamma rhohat`

These are the disc specialisation of the general `Y_perp` source term, and for
`vA_over_U = 0` only `shear` survives, reproducing Kawazura et al. (2022)
equation (2.5): `I_MRI = q (<dbpar d_y psi> - <upar d_y phi>)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmhdgpu.diagnostics.alfvenic import elsasser_energies
from rmhdgpu.diagnostics.budget import flatten_conserved_quantity_budgets
from rmhdgpu.diagnostics.scalar import STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO
from rmhdgpu.diagnostics.spectra import (
    elsasser_perpendicular_spectra,
    perpendicular_shell_spectrum,
)
from rmhdgpu.fourier_diagnostics import modal_average, modal_inner_product_average
from rmhdgpu.operators import dy, dz, inv_lap_perp, lap_perp, poisson_bracket
from rmhdgpu.state import State


EQUATION_SET_NAME = "low_beta_accretion"
FIELD_NAMES = ["psi", "omega", "upar", "dbpar", "drho"]
DEFAULT_INITIAL_CONDITION = "low_beta_accretion_mode"

# Names of the five ideal free-energy source terms, in the order they appear in
# the module docstring. Budget columns are `total_energy_rhs_<name>`.
ENERGY_SOURCE_TERM_NAMES = (
    "shear",
    "curvature",
    "buoyancy",
    "pressure_gradient",
    "entropy_gradient",
)

SCALAR_DIAGNOSTIC_INFO = {
    **STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO,
    "total_energy_rhs_shear": "Signed ideal shear (Reynolds + Maxwell stress) source of d_t total_energy.",
    "total_energy_rhs_curvature": "Signed ideal curvature / grad-B source of d_t total_energy.",
    "total_energy_rhs_buoyancy": "Signed ideal buoyancy (effective gravity) source of d_t total_energy.",
    "total_energy_rhs_pressure_gradient": "Signed ideal background-pressure-gradient source of d_t total_energy.",
    "total_energy_rhs_entropy_gradient": "Signed ideal background-entropy-gradient source of d_t total_energy.",
    "alfvenic_energy": "Alfvenic energy: 0.5 <|grad phi|^2 + |grad psi|^2>.",
    "compressive_energy": "Slow-mode energy: 0.5 <upar^2 + (1 + 1/beta_tilde) dbpar^2>.",
    "entropy_energy": "Entropy-mode energy: 0.5 beta_tilde/(gamma - 1) <(dbpar/beta_tilde + drho)^2>.",
    "upar_energy": "Unweighted parallel kinetic energy: 0.5 <upar^2>.",
    "dbpar_energy": "Weighted compressive magnetic + pressure energy: 0.5 (1 + 1/beta_tilde) <dbpar^2>.",
    "entropy_variance": "Unweighted entropy-variable variance: 0.5 <(dbpar/beta_tilde + drho)^2>.",
    "elsasser_energy_plus": "E+ = 0.5 <|grad(phi - psi)|^2>.",
    "elsasser_energy_minus": "E- = 0.5 <|grad(phi + psi)|^2>.",
    "elsasser_energy_ratio": "Elsasser energy ratio E+ / E-.",
    "normalized_cross_helicity": "(E- - E+) / (E+ + E-) for the package potential convention.",
}


@dataclass(frozen=True, slots=True)
class LowBetaAccretionParameters:
    """Scalar parameters and energy weights used by this equation set.

    This is the first place to edit if a new parameter enters the physics.
    `beta_tilde`, `q`, `mu`, `B_hat`, `P_hat`, `rho_hat` and `gamma` are the six
    inputs; everything else is derived from them.
    """

    beta_tilde: float
    gamma: float
    q: float
    mu: float
    B_hat: float
    P_hat: float
    rho_hat: float
    # Derived combinations appearing in the equations.
    chi: float
    C_b: float
    C_rho: float
    S_hat: float
    alpha_b: float
    alpha_rho: float
    # Energy weights (see `total_energy_modal_density`).
    dbpar_energy_weight: float
    entropy_energy_weight: float


def _param_float(params: Any, name: str) -> float:
    if isinstance(params, Mapping):
        return float(params[name])
    return float(getattr(params, name))


def derived_parameters(params: Any) -> LowBetaAccretionParameters:
    """Return the compact scalar parameter block for this equation set."""

    beta_tilde = _param_float(params, "cs2_over_vA2")
    gamma = _param_float(params, "gamma_ad")
    if beta_tilde <= 0.0:
        raise ValueError(
            "low_beta_accretion requires cs2_over_vA2 > 0 (beta_tilde = cs^2/vA^2); "
            f"got {beta_tilde!r}."
        )
    if gamma <= 1.0:
        raise ValueError(f"low_beta_accretion requires gamma_ad > 1; got {gamma!r}.")

    q = _param_float(params, "q_shear")
    mu = _param_float(params, "vA_over_U")
    B_hat = _param_float(params, "B_hat")
    P_hat = _param_float(params, "P_hat")
    rho_hat = _param_float(params, "rho_hat")

    return LowBetaAccretionParameters(
        beta_tilde=beta_tilde,
        gamma=gamma,
        q=q,
        mu=mu,
        B_hat=B_hat,
        P_hat=P_hat,
        rho_hat=rho_hat,
        # Radial force balance fixes chi = (U^2 - U_K^2)/vA^2 in terms of the
        # background gradients; it is not an independent input.
        chi=beta_tilde * P_hat / gamma + B_hat + 1.0,
        C_b=B_hat - 1.0 - P_hat / gamma,
        C_rho=B_hat - 1.0 + beta_tilde * P_hat / gamma - (1.0 + beta_tilde) * rho_hat,
        # Background entropy gradient d ln(p rho^-gamma) / d ln r.
        S_hat=P_hat - gamma * rho_hat,
        alpha_b=beta_tilde / (1.0 + beta_tilde),
        alpha_rho=1.0 / (1.0 + beta_tilde),
        dbpar_energy_weight=1.0 + 1.0 / beta_tilde,
        entropy_energy_weight=beta_tilde / (gamma - 1.0),
    )


def thin_ring_Lz(params: Any) -> float:
    """Return the normalised parallel box length of one full ring.

    A closed ring has `Lz = 2 pi r`, i.e. `Lz = pi lambda = 2 pi / (vA/U)` in
    the normalised parallel coordinate `z Omega / vA`. A simulation box is
    normally a sub-arc, so `Lz` is chosen independently for resolution (see the
    module docstring); this helper says what the whole-ring length would be, for
    runs that do want to span the full circumference.
    """

    mu = derived_parameters(params).mu
    if mu <= 0.0:
        return float("inf")
    return 2.0 * np.pi / mu


def derive_phi_hat(omega_hat: Any, grid: Any) -> Any:
    """Return `phi_hat = inv_lap_perp(omega_hat)`."""

    return inv_lap_perp(omega_hat, grid)


def derive_j_hat(psi_hat: Any, grid: Any) -> Any:
    """Return `j_hat = -lap_perp(psi_hat) = +k_perp^2 psi_hat`."""

    return -lap_perp(psi_hat, grid)


def derive_entropy_hat(state: State, params: Any) -> Any:
    """Return the entropy variable `sigma = dbpar / beta_tilde + drho`.

    `sigma` is advected but not otherwise coupled to the waves: it satisfies
    `d sigma / dt = -mu Shat / gamma d_y phi`, and it equals `-(1/gamma) ds/c_v`.
    """

    p = derived_parameters(params)
    return state["dbpar"] / p.beta_tilde + state["drho"]


def characteristic_speeds(params: Any) -> list[float]:
    """Return normalised parallel linear speeds relevant to the CFL estimate.

    In these units the Alfven speed is 1 and the slow speed is
    `vs / vA = sqrt(beta_tilde / (1 + beta_tilde))`.
    """

    p = derived_parameters(params)
    return [1.0, float(np.sqrt(p.alpha_b))]


def ideal_rhs(
    state: State,
    grid: Any,
    fft: Any,
    workspace: Any,
    params: Any,
    dealias_mask: Any | None = None,
    out: State | None = None,
) -> State:
    """Return the Fourier-space ideal RHS of the low-beta accretion system.

    `poisson_bracket` is looked up as a module global so the run driver can
    swap in a zero bracket for `equation_mode = "linear"`.
    """

    p = derived_parameters(params)

    psi_hat = state["psi"]
    omega_hat = state["omega"]
    upar_hat = state["upar"]
    dbpar_hat = state["dbpar"]
    drho_hat = state["drho"]

    phi_hat = derive_phi_hat(omega_hat, grid)
    lap_psi_hat = lap_perp(psi_hat, grid)

    rhs_state = state.zeros_like() if out is None else out
    rhs_state.fill_zero()

    # psi: d/dt psi = d_z phi.
    rhs_psi = rhs_state["psi"]
    rhs_psi[...] = dz(phi_hat, grid)
    rhs_psi[...] -= poisson_bracket(phi_hat, psi_hat, grid, fft, workspace, mask=dealias_mask)

    # omega: d/dt omega = nabla_par lap_perp psi
    #                     - 2 d_y upar + 2 mu d_y dbpar - mu chi d_y drho.
    rhs_omega = rhs_state["omega"]
    rhs_omega[...] = dz(lap_psi_hat, grid)
    rhs_omega[...] -= poisson_bracket(phi_hat, omega_hat, grid, fft, workspace, mask=dealias_mask)
    rhs_omega[...] += poisson_bracket(psi_hat, lap_psi_hat, grid, fft, workspace, mask=dealias_mask)
    rhs_omega[...] -= 2.0 * dy(upar_hat, grid)
    rhs_omega[...] += 2.0 * p.mu * dy(dbpar_hat, grid)
    rhs_omega[...] -= p.mu * p.chi * dy(drho_hat, grid)

    # upar: d/dt upar = nabla_par dbpar - mu (Bhat + 1) d_y psi + (2 - q) d_y phi.
    rhs_upar = rhs_state["upar"]
    rhs_upar[...] = dz(dbpar_hat, grid)
    rhs_upar[...] -= poisson_bracket(phi_hat, upar_hat, grid, fft, workspace, mask=dealias_mask)
    rhs_upar[...] += poisson_bracket(psi_hat, dbpar_hat, grid, fft, workspace, mask=dealias_mask)
    rhs_upar[...] -= p.mu * (p.B_hat + 1.0) * dy(psi_hat, grid)
    rhs_upar[...] += (2.0 - p.q) * dy(phi_hat, grid)

    # dbpar: (1 + 1/bt) d/dt dbpar = nabla_par upar + q d_y psi + mu C_b d_y phi,
    # i.e. the wave and source terms carry alpha_b = bt / (1 + bt) = 1 / (1 + 1/bt).
    rhs_dbpar = rhs_state["dbpar"]
    rhs_dbpar[...] = p.alpha_b * dz(upar_hat, grid)
    rhs_dbpar[...] -= poisson_bracket(phi_hat, dbpar_hat, grid, fft, workspace, mask=dealias_mask)
    rhs_dbpar[...] += p.alpha_b * poisson_bracket(psi_hat, upar_hat, grid, fft, workspace, mask=dealias_mask)
    rhs_dbpar[...] += p.alpha_b * p.q * dy(psi_hat, grid)
    rhs_dbpar[...] += p.alpha_b * p.mu * p.C_b * dy(phi_hat, grid)

    # drho: (1 + bt) d/dt drho = -nabla_par upar - q d_y psi - mu C_rho d_y phi,
    # with alpha_rho = 1 / (1 + bt).
    rhs_drho = rhs_state["drho"]
    rhs_drho[...] = -p.alpha_rho * dz(upar_hat, grid)
    rhs_drho[...] -= poisson_bracket(phi_hat, drho_hat, grid, fft, workspace, mask=dealias_mask)
    rhs_drho[...] -= p.alpha_rho * poisson_bracket(psi_hat, upar_hat, grid, fft, workspace, mask=dealias_mask)
    rhs_drho[...] -= p.alpha_rho * p.q * dy(psi_hat, grid)
    rhs_drho[...] -= p.alpha_rho * p.mu * p.C_rho * dy(phi_hat, grid)

    return rhs_state


def linear_matrix(kx: float, ky: float, kz: float, params: Any) -> np.ndarray:
    """Return the 5x5 linear matrix `M` with `d_t f = M f` for one Fourier mode.

    Field order is `[psi, omega, upar, dbpar, drho]`. Terms carrying
    `inv_lap_perp` are set to zero for `k_perp = 0`, where the RMHD potential
    representation is not meaningful (and where `k_y = 0` anyway, so only the
    `psi <- omega` entry actually needs the guard).
    """

    p = derived_parameters(params)
    matrix = np.zeros((5, 5), dtype=np.complex128)

    kx = float(kx)
    ky = float(ky)
    kz = float(kz)
    kperp2 = kx * kx + ky * ky
    inv_kperp2 = 0.0 if kperp2 <= 0.0 else 1.0 / kperp2
    iky = 1j * ky
    ikz = 1j * kz

    # psi
    matrix[0, 1] = -ikz * inv_kperp2
    # omega
    matrix[1, 0] = -ikz * kperp2
    matrix[1, 2] = -2.0 * iky
    matrix[1, 3] = 2.0 * p.mu * iky
    matrix[1, 4] = -p.mu * p.chi * iky
    # upar
    matrix[2, 0] = -p.mu * (p.B_hat + 1.0) * iky
    matrix[2, 1] = -(2.0 - p.q) * iky * inv_kperp2
    matrix[2, 3] = ikz
    # dbpar
    matrix[3, 0] = p.alpha_b * p.q * iky
    matrix[3, 1] = -p.alpha_b * p.mu * p.C_b * iky * inv_kperp2
    matrix[3, 2] = p.alpha_b * ikz
    # drho
    matrix[4, 0] = -p.alpha_rho * p.q * iky
    matrix[4, 1] = p.alpha_rho * p.mu * p.C_rho * iky * inv_kperp2
    matrix[4, 2] = -p.alpha_rho * ikz
    return matrix


def _dissipation_spec_for_field(
    params: Any,
    field_name: str,
    dissipation_spec: Mapping[str, Mapping[str, float | int]] | None,
) -> Mapping[str, float | int]:
    if dissipation_spec is not None:
        return dissipation_spec[field_name]
    if isinstance(params, Mapping):
        return params["dissipation"][field_name]
    return getattr(params, "dissipation")[field_name]


def dissipation_operator(
    grid: Any,
    params: Any,
    field_name: str,
    dissipation_spec: Mapping[str, Mapping[str, float | int]] | None = None,
) -> Any:
    """Return the nonnegative diagonal damping operator `D_i(k)` for one field."""

    spec = _dissipation_spec_for_field(params, field_name, dissipation_spec)
    nu_perp = float(spec["nu_perp"])
    nu_par = float(spec["nu_par"])
    n_perp = int(spec["n_perp"])
    n_par = int(spec["n_par"])

    operator = 0.0
    if nu_perp > 0.0:
        operator = operator + nu_perp * (grid.kperp2**n_perp)
    if nu_par > 0.0:
        operator = operator + nu_par * (grid.kpar2**n_par)
    if isinstance(operator, float):
        operator = grid.kperp2 * 0.0
    return operator


def build_dissipation_operators(
    grid: Any,
    params: Any,
    field_names: list[str] | None = None,
    dissipation_spec: Mapping[str, Mapping[str, float | int]] | None = None,
) -> dict[str, Any]:
    """Build the diagonal damping operators for all evolved fields."""

    names = FIELD_NAMES if field_names is None else field_names
    return {
        name: dissipation_operator(grid, params, name, dissipation_spec=dissipation_spec)
        for name in names
    }


def total_energy_modal_density(state: State, grid: Any, backend: Any, params: Any) -> Any:
    """Return the modal quadratic density for the positive-definite free energy.

    The Alfvenic fields are measured as physical perpendicular amplitudes
    (`u_perp ~ grad_perp phi`, `b_perp ~ grad_perp psi`), so those pieces carry
    `k_perp^2`. In code variables

    `E = 0.5 (|grad_perp phi|^2 + |grad_perp psi|^2 + upar^2`
    `         + (1 + 1/bt) dbpar^2 + bt/(gamma - 1) (dbpar/bt + drho)^2)`.
    """

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    sigma_hat = derive_entropy_hat(state, params)
    return (
        0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
        + 0.5 * xp.abs(state["upar"]) ** 2
        + 0.5 * p.dbpar_energy_weight * xp.abs(state["dbpar"]) ** 2
        + 0.5 * p.entropy_energy_weight * xp.abs(sigma_hat) ** 2
    )


def total_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the volume-averaged free energy for this equation set."""

    return modal_average(total_energy_modal_density(state, grid, backend, params), grid, backend)


def alfvenic_energy(state: State, grid: Any, backend: Any) -> float:
    """Return `0.5 <|grad_perp phi|^2 + |grad_perp psi|^2>`."""

    xp = backend.xp
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = 0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
    return modal_average(density_hat, grid, backend)


def compressive_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the slow-mode energy `0.5 <upar^2 + (1 + 1/bt) dbpar^2>`."""

    xp = backend.xp
    p = derived_parameters(params)
    density_hat = 0.5 * (
        xp.abs(state["upar"]) ** 2 + p.dbpar_energy_weight * xp.abs(state["dbpar"]) ** 2
    )
    return modal_average(density_hat, grid, backend)


def entropy_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the entropy-mode energy `0.5 bt/(gamma - 1) <sigma^2>`."""

    xp = backend.xp
    p = derived_parameters(params)
    sigma_hat = derive_entropy_hat(state, params)
    density_hat = 0.5 * p.entropy_energy_weight * xp.abs(sigma_hat) ** 2
    return modal_average(density_hat, grid, backend)


def _dy_correlation(field_hat: Any, potential_hat: Any, grid: Any, backend: Any) -> float:
    """Return `<field * d_y potential>` for two real fields in rfftn layout."""

    return modal_inner_product_average(field_hat, dy(potential_hat, grid), grid, backend)


def total_energy_source_rhs_terms(
    state: State,
    grid: Any,
    backend: Any,
    params: Any,
) -> dict[str, float]:
    """Return the five signed ideal free-energy source terms.

    These are the disc specialisation of the general `Y_perp` exchange term:
    each entry is the rate at which one background gradient feeds (or drains)
    fluctuation free energy. Their sum is `d_t E` for the ideal, unforced
    system; see the module docstring for the formulae.
    """

    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    psi_hat = state["psi"]
    upar_hat = state["upar"]
    dbpar_hat = state["dbpar"]
    drho_hat = state["drho"]
    sigma_hat = derive_entropy_hat(state, params)

    upar_dy_phi = _dy_correlation(upar_hat, phi_hat, grid, backend)
    upar_dy_psi = _dy_correlation(upar_hat, psi_hat, grid, backend)
    dbpar_dy_phi = _dy_correlation(dbpar_hat, phi_hat, grid, backend)
    dbpar_dy_psi = _dy_correlation(dbpar_hat, psi_hat, grid, backend)
    drho_dy_phi = _dy_correlation(drho_hat, phi_hat, grid, backend)
    sigma_dy_phi = _dy_correlation(sigma_hat, phi_hat, grid, backend)

    return {
        "shear": p.q * (dbpar_dy_psi - upar_dy_phi),
        "curvature": p.mu * (1.0 + p.B_hat) * (dbpar_dy_phi - upar_dy_psi),
        "buoyancy": -p.mu * p.chi * drho_dy_phi,
        "pressure_gradient": -p.mu * p.P_hat / p.gamma * dbpar_dy_phi,
        "entropy_gradient": (
            -p.mu * p.beta_tilde * p.S_hat / (p.gamma * (p.gamma - 1.0)) * sigma_dy_phi
        ),
    }


def total_energy_dissipation_rhs(
    state: State,
    grid: Any,
    backend: Any,
    linear_ops: dict[str, Any],
    params: Any,
) -> float:
    """Return the signed dissipative contribution to `d_t E`.

    Sign convention: `d_t E = sources + dissipation + forcing`, so this is
    negative when damping removes energy.

    `dbpar` and `drho` share the non-diagonal energy block

    `E_c = 0.5 M_bb dbpar^2 + M_br dbpar drho + 0.5 M_rr drho^2`,

    so their damping contributes a cross term. Note that the compressive part
    of the dissipation is guaranteed negative semi-definite only when the two
    fields are damped with the *same* operator (the usual choice); with very
    different `nu` values this block can in principle inject a small amount of
    energy, which is a property of the diagonal-damping model, not a bug.
    """

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)

    m_bb = p.dbpar_energy_weight + p.entropy_energy_weight / (p.beta_tilde**2)
    m_br = p.entropy_energy_weight / p.beta_tilde
    m_rr = p.entropy_energy_weight
    d_b = linear_ops["dbpar"]
    d_r = linear_ops["drho"]

    density_hat = (
        -linear_ops["omega"] * grid.kperp2 * (xp.abs(phi_hat) ** 2)
        - linear_ops["psi"] * grid.kperp2 * (xp.abs(state["psi"]) ** 2)
        - linear_ops["upar"] * xp.abs(state["upar"]) ** 2
        - m_bb * d_b * xp.abs(state["dbpar"]) ** 2
        - m_rr * d_r * xp.abs(state["drho"]) ** 2
        - m_br * (d_b + d_r) * xp.real(state["dbpar"] * xp.conj(state["drho"]))
    )
    return modal_average(density_hat, grid, backend)


def perpendicular_energy_spectra(
    state: State,
    grid: Any,
    backend: Any,
    *,
    bin_width: float | None = None,
    params: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return perpendicular shell spectra for the free-energy pieces.

    `u_perp`, `b_perp`, `upar`, `dbpar` and `s` are the five terms of
    :func:`total_energy_modal_density`, so they sum to `total_energy`. `drho`
    is an extra, *unweighted* `0.5 <|drho|^2>` spectrum: it is not part of the
    energy (the density enters the energy only through the entropy variable
    `s`), but it is useful for seeing where the density fluctuations live.
    """


    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    sigma_hat = derive_entropy_hat(state, params)
    kperp2 = grid.kperp2

    kperp, u_perp = perpendicular_shell_spectrum(
        0.5 * kperp2 * (xp.abs(phi_hat) ** 2), grid, backend, bin_width=bin_width
    )
    _, b_perp = perpendicular_shell_spectrum(
        0.5 * kperp2 * (xp.abs(state["psi"]) ** 2), grid, backend, bin_width=bin_width
    )
    _, upar = perpendicular_shell_spectrum(
        0.5 * (xp.abs(state["upar"]) ** 2), grid, backend, bin_width=bin_width
    )
    _, dbpar = perpendicular_shell_spectrum(
        0.5 * p.dbpar_energy_weight * (xp.abs(state["dbpar"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, entropy = perpendicular_shell_spectrum(
        0.5 * p.entropy_energy_weight * (xp.abs(sigma_hat) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, drho = perpendicular_shell_spectrum(
        0.5 * (xp.abs(state["drho"]) ** 2), grid, backend, bin_width=bin_width
    )
    elsasser = elsasser_perpendicular_spectra(state, grid, backend, bin_width=bin_width)
    return {
        "kperp": kperp,
        "u_perp": u_perp,
        "b_perp": b_perp,
        "upar": upar,
        "dbpar": dbpar,
        "drho": drho,
        "s": entropy,
        "z_plus": elsasser["z_plus"],
        "z_minus": elsasser["z_minus"],
    }


def compute_conserved_quantity_budgets(
    state: State,
    *,
    grid: Any,
    backend: Any,
    params: Any,
    linear_ops: dict[str, Any] | None = None,
    extra_rhs_terms: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the free energy plus its signed RHS budget terms.

    The free energy is not conserved ideally; the five background-gradient
    sources are stored separately with the sign convention

    `d_t E = shear + curvature + buoyancy + pressure_gradient`
    `        + entropy_gradient + dissipation + forcing`.
    """

    rhs_terms: dict[str, float] = total_energy_source_rhs_terms(state, grid, backend, params)
    if linear_ops is not None:
        rhs_terms["dissipation"] = total_energy_dissipation_rhs(
            state, grid, backend, linear_ops, params
        )
    if extra_rhs_terms is not None:
        rhs_terms.update(
            {
                name: float(value)
                for name, value in extra_rhs_terms.get("total_energy", {}).items()
            }
        )

    return {
        "total_energy": {
            "value": total_energy(state, grid, backend, params),
            "rhs_terms": rhs_terms,
        }
    }


def compute_equation_scalar_diagnostics(
    state: State,
    *,
    grid: Any,
    fft: Any,
    backend: Any,
    params: Any,
    workspace: Any | None = None,
    linear_ops: dict[str, Any] | None = None,
    budget_rhs_terms: dict[str, dict[str, float]] | None = None,
    extra_rhs_terms: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Return low-beta-accretion scalar diagnostics plus the energy budget."""

    xp = backend.xp
    p = derived_parameters(params)
    sigma_hat = derive_entropy_hat(state, params)

    alfvenic = alfvenic_energy(state, grid, backend)
    compressive = compressive_energy(state, grid, backend, params)
    entropy = entropy_energy(state, grid, backend, params)
    diagnostics = {
        "alfvenic_energy": alfvenic,
        "compressive_energy": compressive,
        "entropy_energy": entropy,
        "upar_energy": modal_average(0.5 * xp.abs(state["upar"]) ** 2, grid, backend),
        "dbpar_energy": modal_average(
            0.5 * p.dbpar_energy_weight * xp.abs(state["dbpar"]) ** 2, grid, backend
        ),
        "entropy_variance": modal_average(0.5 * xp.abs(sigma_hat) ** 2, grid, backend),
    }
    diagnostics.update(elsasser_energies(state, grid, backend))

    budgets = compute_conserved_quantity_budgets(
        state,
        grid=grid,
        backend=backend,
        params=params,
        linear_ops=linear_ops,
        extra_rhs_terms=extra_rhs_terms,
    )
    rhs_terms = budgets["total_energy"].setdefault("rhs_terms", {})
    if budget_rhs_terms is not None and "total_energy" in budget_rhs_terms:
        rhs_terms.clear()
        rhs_terms.update(
            {name: float(value) for name, value in budget_rhs_terms["total_energy"].items()}
        )
    for name in ENERGY_SOURCE_TERM_NAMES:
        rhs_terms.setdefault(name, 0.0)
    rhs_terms.setdefault("dissipation", 0.0)
    rhs_terms.setdefault("forcing", 0.0)
    diagnostics.update(flatten_conserved_quantity_budgets(budgets))
    return diagnostics
