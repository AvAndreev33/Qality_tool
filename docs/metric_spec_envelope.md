# Metric Batch Spec — Envelope Metrics

## Group-level assumptions

These metrics are defined for a common prepared-signal workflow unless explicitly stated otherwise.

### Default signal recipe
`roi_mean_subtracted_linear_detrended`

### Default recipe binding
`fixed`

### Shared assumptions
- ROI is already correctly extracted around the main packet
- envelope method is fixed globally for the whole batch
- base input before envelope extraction is the prepared signal
- let `e[n]` denote the envelope of that prepared signal
- let `n0 = argmax_n e[n]` and `e_peak = e[n0]`
- shared constants must come from `AnalysisContext`, not from per-metric hardcoded values
- if a required support or side region cannot be determined robustly, the metric should return `valid=False`

### Shared analysis-context parameters
Use these shared names in implementation:
- `analysis_context.epsilon`
- `analysis_context.alpha_main_support`
- `analysis_context.secondary_peak_min_distance`
- `analysis_context.secondary_peak_min_prominence`

If some of these are not yet present in `AnalysisContext`, they should be added there rather than hardcoded in metric code.

---

## Name
`envelope_height`

## Mathematical definition
```text
Input signal: e[n] = envelope of the prepared signal.

1. Main peak:
   e_peak = max_n e[n]

2. Score:
   EH = e_peak
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`envelope`

## Score meaning
`higher is better`

## Notes
- This is the simplest envelope-amplitude metric.
- It depends on the chosen envelope method and on absolute signal scale.
- Metric should return `valid=False` if the envelope is empty or non-finite.

---

## Name
`envelope_area`

## Mathematical definition
```text
Input signal: e[n] = envelope of the prepared signal.

1. Score:
   EA = sum_n e[n]
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`envelope`

## Score meaning
`higher is better`

## Notes
- For uniform sampling, summation in samples is sufficient.
- This metric depends on both envelope height and envelope width.
- Metric should return `valid=False` if the envelope is empty or non-finite.

---

## Name
`envelope_width`

## Mathematical definition
```text
Input signal: e[n] = envelope of the prepared signal.

1. Main peak:
   n0 = argmax_n e[n]
   e_peak = e[n0]

2. Half-maximum level:
   h = 0.5 * e_peak

3. Find left and right crossing positions of e[n] with level h
   around the main peak using linear interpolation:
   z_L, z_R

4. Score:
   EW = z_R - z_L
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`envelope`

## Score meaning
`lower is better`

## Notes
- This is an FWHM-like width metric.
- It measures how localized the main coherence packet is.
- Metric should return `valid=False` if one of the half-maximum crossings does not exist inside ROI.

---

## Name
`envelope_sharpness`

## Mathematical definition
```text
Input signal: e[n] = envelope of the prepared signal.

1. Compute:
   EH = envelope height
   EW = envelope width

2. Score:
   ES = EH / (EW + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`envelope`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- This metric is defined as peak-over-width.
- It favors a high and compact main envelope peak.
- It is intended to be more interpretable and implementation-friendly than curvature-based sharpness.
- Metric should return `valid=False` if envelope width is invalid.

---

## Name
`envelope_symmetry`

## Mathematical definition
```text
Input signal: e[n] = envelope of the prepared signal.

1. Main peak:
   n0 = argmax_n e[n]

2. Define a symmetric comparison range around the peak:
   H = min(n0, M - 1 - n0)

3. Compare mirrored samples:
   D = sum_{m=1..H} |e[n0 - m] - e[n0 + m]|
   S = sum_{m=1..H} (e[n0 - m] + e[n0 + m])

4. Score:
   ESYM = 1 - D / (S + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`envelope`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- This metric measures left-right symmetry of the envelope around the main peak.
- Its ideal range is near `[0, 1]`, with larger values meaning better symmetry.
- Metric should return `valid=False` if no meaningful mirrored comparison range exists.

---

## Name
`single_peakness`

## Mathematical definition
```text
Input signal: e[n] = envelope of the prepared signal.

1. Main peak:
   e_peak = max_n e[n]

2. Define main-peak support:
   W_main = {n : e[n] >= α * e_peak}

3. Score:
   SP = sum_{n in W_main} e[n] / (sum_n e[n] + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`envelope`

## Score meaning
`higher is better`

## Notes
- `α = analysis_context.alpha_main_support`
- `ε = analysis_context.epsilon`
- This metric measures how much of the total envelope mass is concentrated in the main peak region.
- It acts as a practical unimodality-like measure.
- Metric should return `valid=False` if the total envelope mass is near zero or if `W_main` is empty.

---

## Name
`main_peak_to_sidelobe_ratio`

## Mathematical definition
```text
Input signal: e[n] = envelope of the prepared signal.

1. Main peak:
   e_peak = max_n e[n]

2. Define main-peak exclusion region:
   W_main = {n : e[n] >= α * e_peak}

3. Detect local maxima of e[n] outside W_main.

4. Let:
   e_side = maximum height among those secondary maxima
   If none exist, set e_side = 0.

5. Score:
   MPSR = e_peak / (e_side + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`envelope`

## Score meaning
`higher is better`

## Notes
- `α = analysis_context.alpha_main_support`
- `ε = analysis_context.epsilon`
- If secondary peak detection uses explicit thresholds, they should come from:
  - `analysis_context.secondary_peak_min_distance`
  - `analysis_context.secondary_peak_min_prominence`
- This metric measures dominance of the main envelope peak over the strongest secondary structure.
- It is especially useful for double-packet or sidelobe-like cases.

