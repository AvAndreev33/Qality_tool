# Metric Batch Spec — Spectral Metrics

## Group-level assumptions

These metrics are defined for a common prepared-signal workflow unless explicitly stated otherwise.

### Default signal recipe
`roi_mean_subtracted_linear_detrended`

### Default recipe binding
`fixed`

### Shared assumptions
- ROI is already correctly extracted around the main packet
- the prepared signal is additionally Hann-windowed before FFT where the metric definition requires it
- spectral metrics use one-sided positive-frequency spectrum
- DC handling follows the current spectral-layer convention
- expected carrier location is estimated from metadata-derived priors
- shared constants must come from `AnalysisContext`, not from per-metric hardcoded values

### Shared analysis-context parameters
Use these shared names in implementation:
- `analysis_context.epsilon`
- `analysis_context.expected_period_samples`
- `analysis_context.expected_carrier_bin`
- `analysis_context.expected_band_half_width_bins`
- `analysis_context.prominence_window_bins`
- `analysis_context.prominence_exclusion_half_width_bins`
- `analysis_context.local_spectrum_window_cycles`
- `analysis_context.local_spectrum_hop_cycles`

If some of these are not yet present in `AnalysisContext`, they should be added there rather than hardcoded in metric code.

### Shared metadata-derived priors
The following should be computed once from known metadata and placed into the shared context:
- effective oversampling factor
- expected period in samples `T_exp`
- expected carrier bin `k_exp`
- expected working-band half-width `Δk_exp`
- expected band `B_exp`
- lower edge of expected working band `k_low`

These priors should not be recomputed independently inside each metric.

---

## Name
`presence_of_expected_carrier_frequency`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Find strongest bin inside expected carrier band:
   k_* = argmax_{k in B_exp} P[k]

2. Find strongest bin in the whole positive spectrum:
   k_0 = argmax_k P[k]

3. Score:
   PECF = P[k_*] / (P[k_0] + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- `B_exp` must come from shared expected-band priors.
- This metric measures whether the dominant spectral feature is located where the carrier is expected.
- Values near `1` mean the strongest spectral peak lies inside the expected carrier band.
- Metric should return `valid=False` if the expected band is empty or if the usable positive-frequency spectrum is empty.

---

## Name
`dominant_spectral_peak_prominence`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Find dominant spectral peak:
   k_0 = argmax_k P[k]

2. Define local background neighborhood:
   N_loc = {k : |k - k_0| <= W_prom and |k - k_0| > Δk_prom}

3. Score:
   DSPP = P[k_0] / (median_{k in N_loc} P[k] + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`higher is better`

## Notes
- `W_prom = analysis_context.prominence_window_bins`
- `Δk_prom = analysis_context.prominence_exclusion_half_width_bins`
- `ε = analysis_context.epsilon`
- This is a local spectral-contrast metric around the dominant peak.
- Metric should return `valid=False` if `N_loc` is empty.

---

## Name
`carrier_to_background_spectral_ratio`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Compute in-band average power:
   P_car = mean_{k in B_exp} P[k]

2. Compute out-of-band robust background level:
   P_bg = median_{k not in B_exp} P[k]

3. Score:
   CBSR = P_car / (P_bg + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- `B_exp` must come from shared expected-band priors.
- This metric compares the expected carrier band to the global spectral background.
- It is more global than `dominant_spectral_peak_prominence`.
- Metric should return `valid=False` if either the in-band or out-of-band set is empty.

---

## Name
`energy_concentration_in_working_band`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Score:
   ECWB = sum_{k in B_exp} P[k] / (sum_k P[k] + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- `B_exp` must come from shared expected-band priors.
- This is the retained band-energy fraction in the expected working band.
- It is the main global band-concentration metric for this group.
- Metric should return `valid=False` if the usable positive-frequency spectrum is empty.

---

## Name
`low_frequency_trend_energy_fraction`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Define low-frequency region:
   L = {k : 1 <= k < k_low}

2. Score:
   LFTEF = sum_{k in L} P[k] / (sum_k P[k] + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `k_low` must come from shared expected-band priors.
- This metric measures residual low-frequency content below the expected carrier band.
- It is intended to capture incomplete baseline removal or drift leakage.
- Metric should return `valid=False` if the usable positive-frequency spectrum is empty.

---

## Name
`harmonic_distortion_level`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Define harmonic bands, if they lie inside the positive spectrum:
   B_2 = {k : |k - 2*k_exp| <= Δk_exp}
   B_3 = {k : |k - 3*k_exp| <= Δk_exp}

2. Compute harmonic energy:
   E_harm = sum_{k in B_2} P[k] + sum_{k in B_3} P[k]

3. Compute carrier-band energy:
   E_car = sum_{k in B_exp} P[k]

4. Score:
   HDL = E_harm / (E_car + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `k_exp`, `Δk_exp`, and `B_exp` must come from shared expected-band priors.
- This metric uses only the 2nd and 3rd harmonics as the baseline distortion measure.
- If a harmonic band lies outside Nyquist, it is omitted from the sum.
- Metric should return `valid=False` if the carrier band is empty.

---

## Name
`spectral_centroid_offset`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Normalize spectral weights:
   p[k] = P[k] / (sum_k P[k] + ε)

2. Compute spectral centroid:
   μ = sum_k k * p[k]

3. Score:
   SCO = |μ - k_exp| / (k_exp + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `k_exp` must come from shared expected-band priors.
- This metric measures how far the overall spectral center of mass is shifted away from the expected carrier location.
- Metric should return `valid=False` if the usable positive-frequency spectrum is empty.

---

## Name
`spectral_spread`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Normalize spectral weights:
   p[k] = P[k] / (sum_k P[k] + ε)

2. Compute spread around expected carrier:
   SS = sqrt(sum_k (k - k_exp)^2 * p[k]) / (k_exp + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `k_exp` must come from shared expected-band priors.
- This metric measures how broadly spectral energy is dispersed around the expected carrier location.
- Metric should return `valid=False` if the usable positive-frequency spectrum is empty.

---

## Name
`spectral_entropy`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Normalize spectral weights:
   p[k] = P[k] / (sum_k P[k] + ε)

2. Compute normalized entropy:
   SE = -sum_k p[k] * log(p[k] + ε) / log(K)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `K` is the number of usable positive-frequency bins after DC handling.
- Lower values indicate a more concentrated, structured spectrum.
- Higher values indicate a flatter or more noise-like spectrum.
- Metric should return `valid=False` if `K <= 1` or if the usable spectrum is empty.

---

## Name
`spectral_kurtosis`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Normalize spectral weights:
   p[k] = P[k] / (sum_k P[k] + ε)

2. Compute centroid:
   μ = sum_k k * p[k]

3. Compute variance:
   σ^2 = sum_k (k - μ)^2 * p[k]

4. Compute fourth central moment:
   μ_4 = sum_k (k - μ)^4 * p[k]

5. Score:
   SK = μ_4 / (σ^4 + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- This metric quantifies how strongly spectral mass is concentrated into sharp peaks rather than broad distributions.
- It is exploratory and should be interpreted together with other spectral metrics.
- Metric should return `valid=False` if the spectral variance is numerically unstable.

---

## Name
`spectral_peak_sharpness`

## Mathematical definition
```text
Input signal: P[k] = one-sided power spectrum of the prepared ROI signal.

1. Find dominant peak inside expected carrier band:
   k_c = argmax_{k in B_exp} P[k]
   P_c = P[k_c]

2. Define half-maximum level:
   h = 0.5 * P_c

3. Find left and right half-maximum crossing positions
   around k_c using linear interpolation:
   k_L, k_R

4. Peak width:
   W_half = k_R - k_L

5. Score:
   SPS = 1 / (W_half + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power`

## Score meaning
`higher is better`

## Notes
- `ε = analysis_context.epsilon`
- `B_exp` must come from shared expected-band priors.
- This metric measures how narrow the expected carrier peak is.
- Metric should return `valid=False` if half-maximum crossings cannot be found robustly.

---

## Name
`envelope_spectrum_consistency`

## Mathematical definition
```text
Input signals:
- e[n] = envelope of the prepared ROI signal
- P[k] = one-sided power spectrum of the same prepared ROI signal

1. Compute observed envelope width:
   EW_obs = FWHM-like width of e[n]

2. Compute observed spectral spread:
   SS_obs = sqrt(sum_k (k - k_exp)^2 * p[k])
   where p[k] = P[k] / (sum_k P[k] + ε)

3. Build an ideal reference signal using known metadata:
   - mean_wavelength
   - coherence_length
   - oversampling_factor
   - current ROI length M

4. From that ideal reference, compute:
   EW_ref = reference envelope width
   SS_ref = reference spectral spread

5. Define reference width-spread product:
   C_ref = EW_ref * SS_ref

6. Score:
   ESC = |EW_obs * SS_obs - C_ref| / (C_ref + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`envelope+spectrum_power`

## Score meaning
`lower is better`

## Notes
- `ε = analysis_context.epsilon`
- `k_exp` must come from shared expected-band priors.
- This is a cross-representation consistency metric.
- It is intentionally model-dependent and exploratory.
- It requires the same fixed envelope method that is used in the envelope metric batch.
- It also requires metadata-derived ideal reference construction; if that reference cannot be built robustly, the metric should return `valid=False`.

---

## Name
`spectral_correlation_score`

## Mathematical definition
```text
Input signal: x_d[n] = prepared ROI signal before FFT.

1. Define local analysis windows:
   L_win = round(C_win * T_exp)
   H_win = round(C_hop * T_exp)

2. Extract overlapping windows x_j[n] from x_d[n] with hop H_win.

3. For each local window:
   - apply Hann window
   - compute one-sided power spectrum P_j[k]
   - normalize inside the expected local band around the expected local carrier bin

4. For adjacent window pairs, compute normalized spectral correlation:
   ρ_j = corr(q_j, q_{j+1})
   where q_j is the normalized local band-power vector of window j

5. Score:
   SCS = median_j ρ_j
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_power_local`

## Score meaning
`higher is better`

## Notes
- `T_exp = analysis_context.expected_period_samples`
- `C_win = analysis_context.local_spectrum_window_cycles`
- `C_hop = analysis_context.local_spectrum_hop_cycles`
- This is an exploratory cyclostationary-like metric.
- It measures how stable the local spectral structure remains along the ROI.
- It requires local windowed spectra, not only one global spectrum.
- Metric should return `valid=False` if too few local windows are available.