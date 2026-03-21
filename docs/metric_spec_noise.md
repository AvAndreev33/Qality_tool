  # Metric Batch Spec — Noise Metrics

## Group-level assumptions

These metrics are defined for a common prepared-signal workflow unless explicitly stated otherwise.

### Default signal recipe
`roi_mean_subtracted_linear_detrended`

### Default recipe binding
`fixed`

### Shared assumptions
- ROI is already centered on the main packet
- sampling is uniform
- spectral metrics use one-sided positive-frequency spectrum
- DC handling follows the current spectral-layer convention
- shared constants must come from `AnalysisContext`, not from per-metric hardcoded values

### Shared analysis-context parameters
Use these shared names in implementation:
- `analysis_context.epsilon`
- `analysis_context.band_half_width_bins`
- `analysis_context.drift_window`
- `analysis_context.smoothing_window`

If some of these are not yet present in `AnalysisContext`, they should be added there rather than hardcoded in metric code.

---

## Name
`spectral_snr`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Apply Hann window:
   x_w[n] = x_d[n] * w[n]

2. Compute one-sided power spectrum:
   X[k] = rFFT(x_w[n])
   P[k] = |X[k]|^2
   Use only positive-frequency bins excluding DC.

3. Find dominant carrier bin:
   k_c = argmax_k P[k]

4. Define carrier band:
   B = {k : |k - k_c| <= Δk}

5. Compute powers:
   P_signal = sum_{k in B} P[k]
   P_noise  = sum_{k not in B} P[k]

6. Score:
   SNR = 10 * log10((P_signal + ε) / (P_noise + ε))
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
- Use Hann window before FFT.
- `Δk = analysis_context.band_half_width_bins`
- `ε = analysis_context.epsilon`
- Metric should return `valid=False` if the positive-frequency spectrum is empty or if no stable carrier bin can be identified.

---

## Name
`local_snr`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Compute envelope:
   e[n] = envelope(x_d[n])

2. Define main local support:
   e_max = max_n e[n]
   W_sig = {n : e[n] >= 0.5 * e_max}

3. Compute energies:
   E_signal = sum_{n in W_sig} x_d[n]^2
   E_noise  = sum_{n not in W_sig} x_d[n]^2

4. Score:
   LocalSNR = 10 * log10((E_signal + ε) / (E_noise + ε))
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
- Metric depends on envelope quality.
- Metric should return `valid=False` if `W_sig` is empty or if the complementary noise region is empty.

---

## Name
`envelope_peak_to_background_ratio`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Compute envelope:
   e[n] = envelope(x_d[n])

2. Main peak:
   e_peak = max_n e[n]

3. Define exclusion region of the main peak:
   W_main = {n : e[n] >= 0.5 * e_peak}

4. Estimate background level robustly:
   e_bg = median_{n not in W_main} e[n]

5. Score:
   PBR = (e_peak + ε) / (e_bg + ε)
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
- Metric should return `valid=False` if too few samples remain outside `W_main`.

---

## Name
`noise_floor_level`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Apply Hann window:
   x_w[n] = x_d[n] * w[n]

2. Compute one-sided power spectrum:
   X[k] = rFFT(x_w[n])
   P[k] = |X[k]|^2
   Exclude DC.

3. Find dominant carrier bin:
   k_c = argmax_k P[k]

4. Define carrier band:
   B = {k : |k - k_c| <= Δk}

5. Robust out-of-band floor estimate:
   floor = median_{k not in B} P[k]

6. Robust in-band reference:
   ref = median_{k in B} P[k]

7. Score:
   NFL = floor / (ref + ε)
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
- Use Hann window before FFT.
- `Δk = analysis_context.band_half_width_bins`
- `ε = analysis_context.epsilon`
- Metric should return `valid=False` if either the in-band or out-of-band set is empty.

---

## Name
`residual_noise_energy`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Apply Hann window:
   x_w[n] = x_d[n] * w[n]

2. Compute FFT:
   X[k] = FFT(x_w[n])

3. Find dominant carrier bin k_c on positive frequencies and define carrier band:
   B = {k : |k - k_c| <= Δk}
   Keep symmetric negative-frequency bins as well.

4. Build band-limited reconstruction:
   X_B[k] = X[k], if k is in carrier band (and symmetric bins)
   X_B[k] = 0, otherwise

5. Reconstruct useful narrowband component:
   s_hat[n] = real(IFFT(X_B[k]))

6. Residual:
   r[n] = x_w[n] - s_hat[n]

7. Score:
   RNE = sum_n r[n]^2 / (sum_n x_w[n]^2 + ε)
```

## Signal recipe
`roi_mean_subtracted_linear_detrended`

## Recipe binding
`fixed`

## Required representations
`spectrum_complex`

## Score meaning
`lower is better`

## Notes
- Use Hann window before FFT.
- `Δk = analysis_context.band_half_width_bins`
- `ε = analysis_context.epsilon`
- This metric requires complex FFT and symmetric-band reconstruction, not only amplitude or one-sided power spectrum.
- Metric should return `valid=False` if no stable carrier band can be formed.

---

## Name
`high_frequency_noise_level`

## Mathematical definition
```text
Input signal: x_d[n] = prepared signal.

1. Apply Hann window:
   x_w[n] = x_d[n] * w[n]

2. Compute one-sided power spectrum:
   X[k] = rFFT(x_w[n])
   P[k] = |X[k]|^2
   Exclude DC.

3. Find dominant carrier bin:
   k_c = argmax_k P[k]

4. Define carrier band:
   B = {k : |k - k_c| <= Δk}

5. Define high-frequency region:
   H = {k : k > k_c + Δk}

6. Score:
   HFN = sum_{k in H} P[k] / (sum_k P[k] + ε)
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
- Use Hann window before FFT.
- `Δk = analysis_context.band_half_width_bins`
- `ε = analysis_context.epsilon`
- Metric should return `valid=False` if the usable positive-frequency spectrum is empty.

---

## Name
`low_frequency_drift_level`

## Mathematical definition
```text
Input signal: x[n] = ROI signal without detrending.

1. Estimate slow trend by smoothing:
   t[n] = moving_average(x[n], L_drift)

2. Score:
   LFD = sum_n t[n]^2 / (sum_n x[n]^2 + ε)
```

## Signal recipe
`roi_only`

## Recipe binding
`fixed`

## Required representations
`none`

## Score meaning
`lower is better`

## Notes
- Do not subtract mean and do not detrend before computing this metric.
- `L_drift = analysis_context.drift_window`
- `ε = analysis_context.epsilon`
- `L_drift` should be much larger than the fringe period.
- Metric should return `valid=False` if the ROI is too short for a meaningful drift estimate.