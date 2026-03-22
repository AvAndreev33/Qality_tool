# Metric Batch Spec — Regularity Metrics

## Group-level assumptions

These metrics are defined for a common prepared-signal workflow unless explicitly stated otherwise.

### Default signal recipe
`roi_mean_subtracted_linear_detrended`

### Default recipe binding
`fixed`

### Shared assumptions
- ROI is already correctly extracted around the main packet
- sampling along z is uniform
- expected fringe period `T_exp` is known approximately and expressed in samples
- all period- and distance-like quantities are measured in samples
- shared constants must come from `AnalysisContext`, not from per-metric hardcoded values
- if too few valid oscillation cycles are detected, the metric should return `valid=False`

### Shared analysis-context parameters
Use these shared names in implementation:
- `analysis_context.epsilon`
- `analysis_context.expected_period_samples`
- `analysis_context.peak_min_distance_fraction`
- `analysis_context.period_search_tolerance_fraction`
- `analysis_context.cycle_resample_length`

If some of these are not yet present in `AnalysisContext`, they should be added there rather than hardcoded in metric code.

---

## Name
`autocorrelation_peak_strength`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Compute normalized autocorrelation:
   r[τ] = sum_n x_d[n] * x_d[n+τ]
   r_norm[τ] = r[τ] / (r[0] + ε)

2. Define expected-period search interval:
   W_T = {τ : τ_min <= τ <= τ_max}

   where:
   τ_min = round((1 - δ_T) * T_exp)
   τ_max = round((1 + δ_T) * T_exp)

3. Score:
   APS = max_{τ in W_T} r_norm[τ]
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`none`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- `T_exp = analysis_context.expected_period_samples`
- `δ_T = analysis_context.period_search_tolerance_fraction`
- This metric is the main global periodicity metric for the regularity group.
- It should be computed on centered and detrended data; otherwise low-frequency baseline biases the autocorrelation.
- It should return `valid=False` if the expected-period search interval is empty or falls outside the usable lag range.

---

## Name
`local_oscillation_regularity`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Detect local maxima positions:
   p_1, p_2, ..., p_K

   using:
   - minimum peak distance ≈ d_min
   - optional prominence threshold if implemented

2. Define cycles between consecutive maxima:
   cycle i = samples from p_i to p_{i+1}, for i = 1, ..., K-1

3. For each cycle:
   - resample to fixed length L_cycle
   - subtract cycle mean
   - normalize by cycle L2 norm

   Denote normalized cycle waveform as c_i[m], m = 0, ..., L_cycle-1.

4. Compute similarity of neighboring cycles:
   s_i = sum_m c_i[m] * c_{i+1}[m]

   for i = 1, ..., K-2

5. Score:
   LOR = median_i s_i
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`extrema`

## Score meaning
`higher is better`

## Notes
- `d_min = analysis_context.peak_min_distance_fraction * analysis_context.expected_period_samples`
- `L_cycle = analysis_context.cycle_resample_length`
- This metric is defined as cycle-shape consistency, not period stability.
- Per-cycle normalization is essential so that envelope tapering inside ROI does not reduce the score.
- Mark as invalid if fewer than 3 valid cycles are available.

---

## Name
`jitter_of_extrema`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Detect local maxima positions:
   p_1, p_2, ..., p_K

   using:
   - minimum peak distance ≈ d_min
   - optional prominence threshold if implemented

2. Compute inter-peak distances:
   d_i = p_{i+1} - p_i, for i = 1, ..., K-1

3. Compute robust spacing jitter:
   J_ext = MAD(d_i) / (median(d_i) + ε)

   where:
   MAD(d_i) = median(|d_i - median(d_i)|)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`extrema`

## Score meaning
`lower is better`

## Notes
- `d_min = analysis_context.peak_min_distance_fraction * analysis_context.expected_period_samples`
- `ε = analysis_context.epsilon`
- This metric measures cycle spacing stability.
- Use maxima only so the spacing corresponds to one full period.
- A robust statistic such as `MAD / median` is preferred over standard deviation.
- Mark as invalid if fewer than 3 detected maxima are available.

---

## Name
`zero_crossing_stability`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Detect upward zero crossings only:
   crossing between samples n and n+1 exists if:
   x_d[n] < 0 and x_d[n+1] >= 0

2. Estimate crossing positions by linear interpolation:
   z_i = n + (-x_d[n]) / (x_d[n+1] - x_d[n] + ε)

3. Compute distances between consecutive upward crossings:
   d_i = z_{i+1} - z_i

4. Keep only plausible distances near expected period:
   retain d_i such that
   (1 - δ_T) * T_exp <= d_i <= (1 + δ_T) * T_exp

5. Compute robust zero-crossing jitter:
   ZCS = MAD(d_i) / (median(d_i) + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`none`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `T_exp = analysis_context.expected_period_samples`
- `δ_T = analysis_context.period_search_tolerance_fraction`
- Use only upward crossings so the measured spacing corresponds to one full period rather than half-period.
- Linear interpolation makes the metric less quantized and more stable than integer-sample crossing positions.
- This metric is more fragile than extrema-based metrics and should be treated as auxiliary until validated on real data.
- Mark as invalid if too few valid crossings remain after filtering.