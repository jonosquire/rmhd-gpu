"""Branch algebra shared by controlled and stochastic Alfvénic forcing.

The convention is zeta+ = phi - psi, zeta- = phi + psi. The controlled
engine measures native branch vorticity W = lap_perp(zeta), giving branch
energy <|grad zeta|**2>/4. Upstream stochastic epsilon uses twice that energy;
this module changes fields, never the caller's energy normalization.
"""
from __future__ import annotations


def select(array, selection):
    """Read a plane/tile or sparse flat indices without a volume temporary."""
    return array[selection] if isinstance(selection, tuple) else array.reshape(-1)[selection]


def selected_kperp2(grid, selection):
    """Index the perpendicular plane directly; never flatten a broadcast volume."""
    plane = grid.kperp2[..., 0]
    if not isinstance(selection, tuple):
        return plane.reshape(-1)[selection // grid.fourier_shape[-1]]
    values = plane[selection[:2]]
    return values[..., None] if isinstance(selection[2], slice) else values


def alfvenic_fields(config, *, velocity="omega", magnetic="psi"):
    """An equation declares its evolved velocity and magnetic field names."""
    if velocity not in {"phi", "omega"}:
        raise ValueError("Alfvénic forcing velocity must be phi or omega.")
    if velocity not in config.field_names or magnetic not in config.field_names:
        raise ValueError(f"Controlled forcing requires declared fields {velocity!r}, {magnetic!r}.")
    return {"velocity": velocity, "magnetic": magnetic}


def add_potential_increment(state, grid, potential, *, branch, velocity, magnetic="psi",
                            scale=1.0, selection=None):
    """Add delta zeta in exactly one branch (apart from floating roundoff).

    Keep the stochastic path's multiplication order. Controlled forcing passes
    only sparse actuator coefficients and uses the same phi/psi signs.
    """
    if branch not in {"plus", "minus"}:
        raise ValueError("branch must be plus or minus")
    psi_sign = -0.5 if branch == "plus" else 0.5
    if selection is None:
        state[magnetic][...] += psi_sign * scale * potential
        if velocity == "phi":
            state[velocity][...] += (0.5 * scale) * potential
        else:
            state[velocity][...] += (0.5 * scale) * (-grid.kperp2 * potential)
    else:
        state[magnetic].reshape(-1)[selection] += psi_sign * scale * potential
        if velocity == "phi":
            state[velocity].reshape(-1)[selection] += (0.5 * scale) * potential
        else:
            state[velocity].reshape(-1)[selection] += (0.5 * scale) * (-selected_kperp2(grid, selection) * potential)


def standard_metric(config):
    return 1.0


def standard_energy_factors(config):
    return {"plus": 1.0, "minus": 1.0}


def standard_native_parameters(settings, config, branch):
    from rmhdgpu.forcing_control import native_parameters
    basis = settings.target_basis if settings.control == "target" else settings.power_basis
    if basis != "elsasser":
        raise ValueError("This equation supports the elsasser basis, not wave_action.")
    return native_parameters(settings, config, G=1.0)


def standard_branch_values(control, state, branch, selection):
    """Return selected W+ = omega+k²psi or W- = omega-k²psi."""
    k2 = selected_kperp2(control.grid, selection)
    velocity = select(state[control.fields["velocity"]], selection)
    omega = -k2 * velocity if control.fields["velocity"] == "phi" else velocity
    psi = select(state[control.fields["magnetic"]], selection)
    return omega + k2 * psi if branch == "plus" else omega - k2 * psi


def _add_vorticity_increment(control, state, branch, increment):
    potential = -increment / selected_kperp2(control.grid, control.indices)
    add_potential_increment(state, control.grid, potential, branch=branch,
                            velocity=control.fields["velocity"], magnetic=control.fields["magnetic"],
                            selection=control.indices)


def standard_apply_gain(control, state, branch, factor):
    before = standard_branch_values(control, state, branch, control.indices)
    _add_vorticity_increment(control, state, branch, (factor - 1) * before)


def standard_seed_branch(control, state, branch, values):
    before = standard_branch_values(control, state, branch, control.indices)
    _add_vorticity_increment(control, state, branch, values - before)


def standard_characteristic_speed(config, branch):
    return config.vA if branch == "plus" else -config.vA


def standard_budget_work(event):
    return {"total_energy": event["work_z"]}
