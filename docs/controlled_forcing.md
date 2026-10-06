# Controlled shell forcing

[Back to the README](../README.md#controlled-shell-forcing)

The controller drives a physical Alfvénic branch amplitude towards a specified
target, or requests a constant energy input per unit time. The equations still
evolve normally between control events. The implementation supports `alfvenic`,
`s09`, and `low_beta_stratified`; all use the same controller functions.

## Run an example

```sh
python -m rmhdgpu.run examples/controlled_target_s09.input --backend cupy
python -m rmhdgpu.run examples/controlled_power_alfvenic.input --backend cupy
```

Omit the override to use SciPy on a CPU. The examples write to separate
directories beneath `examples/outputs/`, with full-field output disabled.
The [S09 input](../examples/controlled_target_s09.input) uses the full nonlinear
RHS with only the plus branch initially populated. There is no reflection or
compressive perturbation, so nonlinear self-interaction vanishes analytically.
It follows target relaxation for seven Alfvén crossing times. The
[balanced-power input](../examples/controlled_power_alfvenic.input) is a short
nonlinear energy-budget check, not a calibrated turbulence run.

## Branches and normalization

The repository's potential convention is

$$\zeta^+=\phi-\psi,\qquad \zeta^-=\phi+\psi.$$

The physical perpendicular vector has magnitude $|\nabla_\perp\zeta^\pm|$.
Controlled branch energy and RMS are

$$E_z^\pm=\frac14\langle|\nabla_\perp\zeta^\pm|^2\rangle,
\qquad z^\pm_{\rm rms}=2\sqrt{E_z^\pm}.$$

The target is a vector amplitude, not the RMS of the stored potential or
vorticity. `target_quantity="rms"` converts the specified value using
$E_*=(z_{{\rm rms},*})^2/4$; `"energy"` specifies $E_*$ directly.
These standard equations accept only `target_basis="elsasser"` or
`power_basis="elsasser"`. The portable engine's wave-action units require an
equation-specific conversion and are rejected here.

The existing `elsasser_energy_plus/minus` diagnostics and stochastic
`epsilon_plus/minus` use twice this branch energy. Their normalization is
unchanged. Thus a controlled power of $\epsilon_c$ corresponds to a stochastic
branch power of $2\epsilon_c$ in mean injection. For controlled forcing,
the total Alfvénic energy receives the sum of the two branches' actual work.
The historical `_f` controller columns mean native branch energy; for these
standard equations they equal the corresponding `_z` columns.

## Actuator and measurement scopes

`k_sigma_min/max` select a band in physical perpendicular wavenumber
$k_\perp=\sqrt{k_x^2+k_y^2}$ for these equations. They are not integer shell
numbers when box lengths differ from $2\pi$. `kz_index` is a positive integer
harmonic, $k_z=2\pi\,\mathtt{kz\_index}/L_z$; it must exclude zero and Nyquist.
The real field contains its conjugate negative harmonic as well. At least
eight stored actuator modes are required, and every actuator mode must survive
the selected dealias mask.

All scopes change only this actuator. They differ in what energy is measured:

| `target_scope` | Feedback measurement |
|---|---|
| `branch_total` (default) | Entire branch, including $k_z=0$ |
| `branch_nonzero_kz` | Entire branch excluding $k_z=0$ |
| `shell` | Only the actuator band at the selected parallel harmonic |
| `perpendicular_shell` | The same perpendicular band across all retained parallel modes |

Selecting a measurement scope does not remove modes from the equations.
For example, excluding $k_z=0$ from feedback does not project it out of the state.

## One control event

Let $E_a$ be actuator energy, $E_c$ the measured energy and $h$ the elapsed
interval since the previous event. For target control,

$$r=1-e^{-h/\tau_F},\qquad
\Delta W=E_c\left[\exp\left(r\log(E_*/E_c)\right)-1\right].$$

Constant-power control instead requests $\Delta W=\epsilon h$. The actuator's
logarithmic amplitude gain is

$$\ell=\frac12\log(1+\Delta W/E_a).$$

For broad feedback scopes, negative work cannot drain the actuator below its
configured floor. The amplitude-growth rate $\ell/h$ is then capped by
`gamma_max`. Consequently achieved amplitude or power can differ from the
requested value. Floor and cap activity are reported explicitly. During
evolution an empty actuator is an error: multiplicative forcing cannot seed it.

The multiplier is $e^\ell$ for `phase_model="none"`. For
`"fixed_gain_angle"`, it is $\exp[\ell(1-i\tan\theta_k)]$, with mode angles
fixed at startup. Separate random streams preserve initial templates when the
phase option changes. This is not white-in-time noise.

Events occur after completed PDE steps. Actual accepted timesteps accumulate
until the forcing interval is reached; forcing does not shorten a PDE step.
The next requested interval is bounded by `interval_max` and
`log_gain_target/max(abs(gamma_applied))`. There is at most one event per PDE
step, and the last step flushes any pending interval.

## Defaults and startup

With $\tau_A=L_z/v_A$ and $\omega_A=k_zv_A$:

| Setting | Default |
|---|---|
| `tau_F` | $\tau_A$ |
| `interval_max` | $0.1/\omega_A$ |
| `gamma_max`, target / constant power | $\omega_A$ / $10\omega_A$ |
| Target `startup_energy_fraction` | 0.01 of target energy |
| Broad-scope `ring_floor_fraction` | 0.01 of target energy |
| Power `startup_energy_factor` | 10, giving startup energy $10\epsilon\tau_A$ |
| `log_gain_target`, `log_gain_warn` | 0.05, 0.1 |
| `phase_model`, `phase_angle_max` | `none`, 0.5 radians |

Startup adds energy only when necessary: it rescales a populated actuator or
uses its seeded template when empty. Broad-scope target startup also respects
energy already outside the actuator. It does not reset an existing state to
the target. Startup work is recorded separately from subsequent kick work.

For an unloaded branch without caps, floors or numerical damping, the response is

$$\log\frac{E_*}{E(t)}=\log\frac{E_*}{E(0)}e^{-t/\tau_F}.$$

This is asymptotic relaxation; even a correct finite-duration run generally
finishes below its RMS target when started below it.

## Inputs and output

Set `[forcing] type="controlled_shell"`, `use_forcing=true`, an explicit
`branches=["plus"]`, `["minus"]` or `["plus", "minus"]`, and one table per
selected branch. Unselected branches receive no controlled kick. They can
still evolve through the equations. Target and constant-power settings are
mutually exclusive within each branch. See the two complete inputs above.

Stochastic settings such as `forcing_mode`, `epsilon_plus`, and
`field_energy_injection_rates` cannot be mixed into a controlled-shell input.
Old `force_amplitudes` syntax and `--force-sigma` are rejected rather than
reinterpreted as powers.

The normal driver writes:

- `forcing_metadata.json`: resolved settings, units, startup work and definitions;
- `forcing_modes.csv`: the exact actuator, Fourier weights and phase angles;
- `forcing_events.csv`: intervals, gains, requested and actual signed work,
  cap/floor activity and measured energies;
- `forcing_*` columns in `scalar_diagnostics.csv`: current amplitude, cumulative
  work, startup work, event count and pending forcing time.

Actual work is measured as post-kick minus pre-kick actuator energy, including
the quadratic contribution. `total_energy_rhs_forcing` reports this signed
work averaged over the scalar-output interval. Startup is already present in
the first state and is not added again to subsequent forcing work.

## Implementation and another equation set

The procedural algorithm is in [forcing_control.py](../rmhdgpu/forcing_control.py).
Small dataclasses hold settings and state. The
[controlled_forcing.py](../rmhdgpu/controlled_forcing.py) facade delegates to
these functions; it contains no second control algorithm.
[forcing_fields.py](../rmhdgpu/forcing_fields.py) provides shared branch algebra
and bounded-memory energy reductions. Equation modules explicitly declare
their hooks; see the end of [s09.py](../rmhdgpu/equations/s09.py).

| Hooks | Responsibility |
|---|---|
| `forcing_fields`, `forcing_metric` | Stored fields and perpendicular geometry |
| `forcing_native_parameters`, `forcing_energy_factors`, `forcing_characteristic_speed` | Units and characteristic scales |
| `forcing_branch_values`, `forcing_apply_gain`, `forcing_seed_branch` | Read and change branch coefficients |
| `forcing_shell_density`, `forcing_perpendicular_energy`, `forcing_perpendicular_shell_energy`, `forcing_measurement` | Energy and feedback measurements |
| `forcing_budget_work` | Convert actual kick work into equation budget terms |

For standard potential storage, declare
`alfvenic_fields(config, velocity="phi", magnetic="psi")`; for vorticity
storage use `velocity="omega"`. A plus increment $d=\delta\zeta^+$ applies
$\delta\phi=d/2$, $\delta\psi=-d/2$, and
$\delta\omega=\nabla_\perp^2d/2$. The minus increment reverses the magnetic
sign. This preserves an existing opposite branch up to floating-point roundoff.
Other evolved fields are unchanged by the actuator.

An additional equation with this algebra can reuse the standard hooks. Its
energy, physical field convention and RHS still need independent verification.
Unsupported equations fail explicitly. Core serialization is available, but
this port does not add a checkpoint/restart driver to original rmhd-gpu.

## Validation

Run focused checks with:

```sh
python -m pytest -q rmhdgpu/tests/test_forcing_control.py \
  rmhdgpu/tests/test_standard_controlled_forcing.py \
  rmhdgpu/tests/test_forcing.py rmhdgpu/tests/test_runfile_parser.py
python -m rmhdgpu.examples.controlled_forcing_checks --backend cupy --output output/controlled_checks
```

Use a compute allocation on Aoraki. The example checker runs the two supplied
inputs with their fixed endpoints, checks target relaxation, branch isolation,
normalization, signed work and budgets, and writes a compact report. The tests
cover both field representations with an already populated opposite branch,
all feedback scopes, serialization, both evolution drivers, stochastic forcing
and input round trips. GPU tests skip cleanly if CUDA is unavailable.

The portable numerical source was first validated in rmhd-gpu-squish at
`b79f3e3`: 451 focused checks on an A100, five complete bitwise legacy trajectory
comparisons and cross-source continuation passed. An isolated original-repository
port preview passed 51 CPU tests. That history supports the shared algorithm;
the assembled upstream branch is checked separately using the commands above.
Detailed job products stay outside Git. No production campaign is part of this port.
