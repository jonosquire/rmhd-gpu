"""Compatibility API for the procedural controlled-forcing implementation.

Algorithms live in forcing_control; equation hooks own field representations.
Existing drivers/checkpoints may continue to use ControlledShellForcing.
"""
from typing import Any
from pathlib import Path
from rmhdgpu import forcing_control as core
from rmhdgpu.forcing_control import (
    ControlledBranchSettings, ControlledShellSettings, control_event_map, settings_document,
)


def _equation(config):
    from rmhdgpu.equations import get_equation_module
    equation = get_equation_module(config.equation_set)
    if not callable(getattr(equation, "forcing_fields", None)):
        raise ValueError("controlled_shell forcing only supports equations with declared forcing hooks.")
    return equation


def native_branch_parameters(settings, config, branch):
    return _equation(config).forcing_native_parameters(settings, config, branch)


def controlled_shell_geometry(config, settings):
    return core.shell_geometry(config, settings, sigma=_equation(config).forcing_metric(config))


class ControlledShellForcing(core.ForcingState):
    """Thin method facade; all state transitions are explicit core functions."""

    def __init__(self, config, grid, backend, dealias_mask=None):
        super().__init__(config, grid, backend, _equation(config))
        core.configure(self, config, grid, backend, dealias_mask)

    def shell_density(self, state: Any, branch: str):
        return core.shell_density(self, state, branch)

    def shell_energy(self, state: Any, branch: str):
        return core.shell_energy(self, state, branch)

    def perpendicular_energy(self, state: Any, branch: str, *, nonzero_kz: bool=False):
        return core.perpendicular_energy(self, state, branch, nonzero_kz=nonzero_kz)

    def branch_energy(self, state: Any, branch: str):
        return core.branch_energy(self, state, branch)

    def perpendicular_shell_energy(self, state: Any, branch: str):
        return core.perpendicular_shell_energy(self, state, branch)

    def _energy_measurement(self, state: Any, branch: str):
        return core._energy_measurement(self, state, branch)

    def initialize(self, state: Any):
        return core.initialize(self, state)

    def advance(self, state: Any, dt: float, time: float, *, final: bool=False):
        return core.advance(self, state, dt, time, final=final)

    def diagnostics(self, state: Any):
        return core.diagnostics(self, state)

    def write_metadata(self, output_dir: str | Path):
        return core.write_metadata(self, output_dir)


def build_controlled_forcing(config: Any, state: Any, dealias_mask: Any | None = None, *, initialize: bool = True) -> ControlledShellForcing | None:
    if not getattr(config, "use_forcing", False) or getattr(config, "forcing_type", "stochastic") != "controlled_shell":
        return None
    controller = ControlledShellForcing(config, state.grid, state.backend, dealias_mask)
    if initialize:
        controller.initialize(state)
    return controller
