# Metric Batch Spec — Phase Metrics

## Group-level assumptions

These metrics are defined for a common prepared-signal workflow unless explicitly stated otherwise.

### Default signal recipe
`roi_mean_subtracted_linear_detrended`

### Default recipe binding
`fixed`

### Shared assumptions
- ROI is already correctly extracted around the main coherence packet.
- Optional smoothing is allowed only if it is zero-phase.
- Phase metrics are defined only inside the useful packet support, not on the full ROI.
- The envelope and analytic phase must be computed from the same prepared signal.
- If physical `z_axis` exists, all phase-derivative quantities must be computed with respect to `z`; otherwise with respect to sample index.
- Edge regions of the analytic signal are considered unreliable and must be excluded by a fixed guard trimming.
- Shared constants must come from `AnalysisContext`, not from per-metric hardcoded values.
- If the phase support cannot be determined robustly, or too few valid samples remain, the metric should return `valid=False`.

### Shared derived representations
Let:
- `x_p[n]` = prepared ROI signal
- `a[n] = x_p[n] + i H{x_p[n]}` = analytic signal
- `e[n] = |a[n]|` = envelope
- `φ[n] = unwrap(angle(a[n]))` = unwrapped phase
- `n0 = argmax_n e[n]`
- `e_peak = e[n0]`

Define phase-support candidates:
- `S0 = { n : e[n] >= α_phase * e_peak }`

Then:
- keep only the largest connected support containing `n0`
- trim `g_phase` samples from both support edges
- denote the final support by `S`

Define local coordinate:
- `u[n] = z[n]`, if physical `z_axis` exists
- otherwise `u[n] = n`

Define local phase slope on adjacent valid support samples:
- `d[i] = (φ[i+1] - φ[i]) / (u[i+1] - u[i] + ε)`

### Shared analysis-context parameters
Use these shared names in implementation:
- `analysis_context.epsilon`
- `analysis_context.phase_support_threshold_fraction`
- `analysis_context.phase_guard_samples`
- `analysis_context.phase_weight_power`
- `analysis_context.phase_monotonicity_tolerance_fraction`
- `analysis_context.phase_jump_tolerance_fraction`
- `analysis_context.minimum_phase_support_samples`
- `analysis_context.minimum_phase_support_periods`

If some of these are not yet present in `AnalysisContext`, they should be added there rather than hardcoded in metric code.

### Shared validity rules
A phase metric should return `valid=False` if any of the following holds:
- the support `S` cannot be determined robustly
- fewer than `minimum_phase_support_samples` remain in `S`
- the support contains too few apparent carrier periods
- the median local phase slope is numerically too close to zero
- analytic phase becomes unstable on most of the support

---

## Name
`phase_slope_stability`

## Mathematical definition
Input signal: `φ[n] = unwrapped phase on support S`

1. Compute local phase slopes:
   `d[i] = (φ[i+1] - φ[i]) / (u[i+1] - u[i] + ε)`

2. Compute robust central slope:
   `d_med = median_i d[i]`

3. Compute robust spread:
   `spread = MAD_i(d[i])`

   where:
   `MAD_i(d[i]) = median_i(|d[i] - d_med|)`

4. Score:
   `PSS = spread / (|d_med| + ε)`

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Uses
`analytic_phase`, `envelope`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- This is the main local phase-stability metric for the group.
- It measures how constant the local carrier slope remains inside the useful packet.
- If `|d_med|` is too small for stable normalization, the metric should be invalid.

---

## Name
`phase_linear_fit_residual`

## Mathematical definition
Input signal: `φ[n] = unwrapped phase on support S`

1. Define weights:
   `w[n] = (e[n] / (e_peak + ε))^p`

   where:
   `p = phase_weight_power`

2. Fit weighted linear model:
   `φ[n] ≈ β0 + β1 * u[n]`

3. Compute residual:
   `r[n] = φ[n] - (β0 + β1 * u[n])`

4. Score:
   `PLFR = sqrt( sum_{n in S} w[n] * r[n]^2 / (sum_{n in S} w[n] + ε) ) / π`

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Uses
`analytic_phase`, `envelope`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `p = analysis_context.phase_weight_power`
- Envelope weighting suppresses low-amplitude tails where phase is less reliable.
- This is the main global phase-shape metric in the group.
- It is sensitive to unwrap failures, local distortions, and non-affine phase behavior.

---

## Name
`phase_curvature_index`

## Mathematical definition
Input signal: `φ[n] = unwrapped phase on support S`

1. Define weights:
   `w[n] = (e[n] / (e_peak + ε))^p`

2. Fit weighted quadratic model:
   `φ[n] ≈ γ0 + γ1 * u[n] + γ2 * u[n]^2`

3. Define support span:
   `L = max_{n in S} u[n] - min_{n in S} u[n]`

4. Score:
   `PCI = |γ2| * L / (|γ1| + ε)`

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Uses
`analytic_phase`, `envelope`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `p = analysis_context.phase_weight_power`
- This metric measures deviation from approximately constant local carrier frequency across the packet.
- It is research-interesting and may reflect dispersion-like or chirp-like behavior, not only generic signal degradation.
- Interpretation should therefore be more cautious than for `phase_slope_stability` or `phase_linear_fit_residual`.

---

## Name
`phase_monotonicity_score`

## Mathematical definition
Input signal: `φ[n] = unwrapped phase on support S`

1. Compute local phase slopes:
   `d[i] = (φ[i+1] - φ[i]) / (u[i+1] - u[i] + ε)`

2. Compute robust reference slope:
   `d_med = median_i d[i]`
   `s_ref = sign(d_med)`

3. Define monotone inlier set:
   `M = { i : s_ref * d[i] > 0 and |d[i] - d_med| <= τ_mon * |d_med| }`

4. Define pair weights:
   `w[i] = min(e[i], e[i+1]) / (e_peak + ε)`

5. Score:
   `PMS = sum_{i in M} w[i] / (sum_i w[i] + ε)`

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Uses
`analytic_phase`, `envelope`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- `τ_mon = analysis_context.phase_monotonicity_tolerance_fraction`
- This is a robust sanity metric for whether phase evolves mostly in one direction with limited local deviations.
- It should be interpreted together with `phase_slope_stability`, not as a standalone phase-quality metric.

---

## Name
`phase_jump_fraction`

## Mathematical definition
Input signal: `φ[n] = unwrapped phase on support S`

1. Compute local phase slopes:
   `d[i] = (φ[i+1] - φ[i]) / (u[i+1] - u[i] + ε)`

2. Compute robust reference slope:
   `d_med = median_i d[i]`

3. Define jump set:
   `J = { i : sign(d[i]) != sign(d_med) or |d[i] - d_med| > τ_jump * |d_med| }`

4. Score:
   `PJF = |J| / (number_of_valid_slopes + ε)`

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Uses
`analytic_phase`, `envelope`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `τ_jump = analysis_context.phase_jump_tolerance_fraction`
- This metric is primarily diagnostic.
- It is useful for unwrap pathology, packet fragmentation, or local phase collapse.
- It should not be treated as the main phase score for the group.

---

## Recommended group interpretation

For baseline use in the GUI and routine analysis, the primary phase metrics should be:
- `phase_slope_stability`
- `phase_linear_fit_residual`
- `phase_curvature_index`
- `phase_monotonicity_score`

`phase_jump_fraction` should be kept as a diagnostic or exploratory metric rather than a primary phase-quality score.