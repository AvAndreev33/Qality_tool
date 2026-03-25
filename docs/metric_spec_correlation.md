# Metric Batch Spec — Correlation / Reference-Model Metrics

## Group-level assumptions

These metrics are defined for a common prepared-signal workflow and a common simple reference model unless explicitly stated otherwise.

### Default signal recipe
`roi_mean_subtracted_linear_detrended`

### Default recipe binding
`fixed`

### Shared assumptions
- ROI is already correctly extracted around the main coherence packet.
- ROI is assumed to be centered on the main raw-signal maximum.
- These metrics use a fixed simple reference model: centered Gaussian envelope × cosine carrier.
- The reference model is built with the same length as the current ROI.
- Amplitude scale and DC level are intentionally excluded from comparison.
- Model comparison is therefore performed on detrended and normalized signals.
- Physical-scale consistency is mandatory for this group:
  - the reference carrier period must be expressed in the same axis units as the ROI axis,
  - the reference envelope scale must be expressed in the same axis units as the ROI axis.
- Conversion from acquisition metadata (`wavelength_nm`, `coherence_length_nm`, `z_step_nm`, acquisition geometry) to usable reference constants must be performed once in shared context, not inside each metric.
- If required metadata or physical scaling is unavailable, these metrics should return `valid=False`.

### Shared derived representations
Let:
- `x_d[n]` = processed ROI signal
- `M` = ROI length
- `n_c = (M - 1) / 2`

Define centered physical coordinate:
- if physical `z_axis` exists and is valid:
  `u[n] = z_axis[n] - z_axis[n_c]`
- otherwise:
  `u[n] = (n - n_c) * z_step_nm`

Define reference constants:
- `T_ref_nm` = expected carrier period in ROI-axis units
- `L_ref_nm` = expected Gaussian envelope 1/e scale in ROI-axis units

These must be pre-derived once from metadata and acquisition geometry.

Define centered reference envelope:
- `g_ref[n] = exp(-(u[n] / L_ref_nm)^2)`

Define centered quadrature reference pair:
- `r_c[n] = g_ref[n] * cos(2π * u[n] / T_ref_nm)`
- `r_s[n] = g_ref[n] * sin(2π * u[n] / T_ref_nm)`

Define reference support:
- `S_ref = { n : g_ref[n] >= α_ref }`

where:
- `α_ref = analysis_context.reference_support_threshold_fraction`

Define zero-mean unit-norm normalization on `S_ref`:
- for any vector `v[n]`, restricted to `S_ref`,
  `ṽ = (v - mean(v)) / (sqrt(sum(v - mean(v))^2) + ε)`

Define normalized observed signal:
- `x̃ = normalized version of x_d on S_ref`

Define normalized fixed-phase reference:
- `c̃ = normalized version of r_c on S_ref`

Define orthonormal phase-flexible reference basis:
1. `q1 = c̃`
2. `s̃ = normalized version of r_s on S_ref`
3. `q2_raw = s̃ - <s̃, q1> * q1`
4. `q2 = q2_raw / (||q2_raw|| + ε)`

where `<a, b>` denotes inner product on `S_ref`.

### Shared envelope representation
For envelope-based metrics in this group:
- compute observed envelope `e_obs[n]` from the same processed signal `x_d[n]`
- use the globally selected fixed envelope method for the whole batch
- compare envelope shape only on `S_ref`

### Shared analysis-context parameters
Use these shared names in implementation:
- `analysis_context.epsilon`
- `analysis_context.reference_support_threshold_fraction`
- `analysis_context.reference_carrier_period_nm`
- `analysis_context.reference_envelope_scale_nm`
- `analysis_context.minimum_reference_support_samples`

If some of these are not yet present in `AnalysisContext`, they should be added there rather than hardcoded in metric code.

### Shared validity rules
A correlation/reference metric should return `valid=False` if any of the following holds:
- required physical scaling is unavailable
- `T_ref_nm <= 0` or `L_ref_nm <= 0`
- support `S_ref` cannot be determined robustly
- fewer than `minimum_reference_support_samples` remain in `S_ref`
- normalized reference basis becomes numerically unstable
- observed signal energy on `S_ref` is numerically too small

---

## Name
`centered_reference_correlation`

## Mathematical definition
Input signal: `x_d[n]` = processed ROI signal.

1. Build centered fixed-phase reference:
   `r_c[n] = g_ref[n] * cos(2π * u[n] / T_ref_nm)`

2. Restrict both `x_d[n]` and `r_c[n]` to `S_ref`.

3. Normalize both on `S_ref`:
   - `x̃ = normalize(x_d)`
   - `c̃ = normalize(r_c)`

4. Score:
   `CRC = <x̃, c̃>`

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Uses
`reference_model`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- This is the strictest metric in the group.
- It tests similarity to the exact centered reference model with fixed carrier phase.
- It intentionally rewards both correct packet shape and correct phase placement relative to ROI center.
- It is sensitive to suboptimal centering and to phase mismatch at the ROI center.

---

## Name
`best_phase_reference_correlation`

## Mathematical definition
Input signal: `x_d[n]` = processed ROI signal.

1. Build centered quadrature reference pair:
   - `r_c[n] = g_ref[n] * cos(2π * u[n] / T_ref_nm)`
   - `r_s[n] = g_ref[n] * sin(2π * u[n] / T_ref_nm)`

2. Restrict to `S_ref`.

3. Normalize and orthonormalize:
   - `x̃ = normalize(x_d)`
   - `q1 = normalized r_c`
   - `q2 = orthonormalized version of normalized r_s` against `q1`

4. Compute projection coefficients:
   - `a1 = <x̃, q1>`
   - `a2 = <x̃, q2>`

5. Score:
   `BPRC = sqrt(a1^2 + a2^2)`

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Uses
`reference_model`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- This is the main similarity metric for the group.
- It measures how well the observed ROI matches the expected packet shape when trivial carrier-phase offset is allowed.
- It is less brittle than `centered_reference_correlation`.
- It should be treated as the primary model/correlation score for baseline use.

---

## Name
`reference_envelope_correlation`

## Mathematical definition
Input signal: `x_d[n]` = processed ROI signal.

1. Compute observed envelope from the same processed signal:
   `e_obs[n] = envelope(x_d[n])`

2. Build centered reference envelope:
   `g_ref[n] = exp(-(u[n] / L_ref_nm)^2)`

3. Restrict both `e_obs[n]` and `g_ref[n]` to `S_ref`.

4. Normalize both on `S_ref`:
   - `ẽ = normalize(e_obs)`
   - `ĝ = normalize(g_ref)`

5. Score:
   `REC = <ẽ, ĝ>`

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Uses
`reference_model+envelope`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- This metric compares only packet-shape similarity, not carrier-phase agreement.
- It is useful for separating envelope mismatch from carrier-phase mismatch.
- It depends on the fixed global envelope method used in the batch.

---

## Name
`phase_relaxation_gain`

## Mathematical definition
Input signal: same as for `centered_reference_correlation` and `best_phase_reference_correlation`.

1. Compute:
   - `CRC = centered_reference_correlation`
   - `BPRC = best_phase_reference_correlation`

2. Score:
   `PRG = BPRC - CRC`

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Uses
`reference_model`

## Score meaning
`lower is better`

## Notes
- This is a diagnostic metric.
- Small values mean that freeing the carrier phase does not improve model agreement much.
- Large values mean that the packet shape may still be reasonable, but the strict centered-phase assumption is violated.
- This can indicate imperfect ROI centering, local phase shift, or model phase mismatch.

---

## Recommended group interpretation

For baseline use, the primary metrics in this group should be:
- `best_phase_reference_correlation`
- `reference_envelope_correlation`
- `centered_reference_correlation`

`phase_relaxation_gain` should be treated as a diagnostic companion metric rather than the main quality score.

### Practical interpretation
- high `BPRC` + high `REC` + high `CRC`  
  signal matches the simple centered reference model well

- high `BPRC` + high `REC` + low `CRC`  
  packet shape is good, but strict center-phase agreement is poor

- low `BPRC` + high `REC`  
  envelope shape is acceptable, but carrier behavior is inconsistent with the model

- low `REC`  
  packet shape itself does not agree well with the expected Gaussian reference envelope