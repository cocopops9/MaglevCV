# 7 · Results and limitations

[← 6 · Kalman filter](06-kalman-filter.md) · [Guide index](README.md)

## 7.1 Results on `measured_rotation.csv`

| Stage | Quantity | Value |
|-------|----------|-------|
| Measurement | samples, rate, resolution | 5926, 30 Hz, 1° |
| Naive fit | SSR / RMS residual | 1,084,427 deg² / 13.5° (no oscillation captured) |
| Simple fit | SSR / RMS residual | 715,619 deg² / 11.0° |
| Simple fit | $\beta$, $\omega$, offset | 0.0297 s⁻¹, 3.732 rad/s (period 1.68 s), 49.8° |
| Advanced fit | SSR / RMS residual | 300,861 deg² / 7.1° |
| Advanced fit | $\beta_0,\ \beta_1,\ \omega_0,\ \omega_1$ | −0.0113 s⁻¹, 1.83·10⁻⁴ s⁻², 3.541 rad/s, 4.12·10⁻³ rad/s² |
| Kalman (as written) | $\sigma_z$, $\sigma_q^{\star}$, log L | 7.13°, 0.494, 4661.6 |
| Kalman (as written) | RMSE estimate vs measurement, innovation std | 2.69°, 4.52° |
| Kalman (textbook update, for comparison) | $\sigma_q^{\star}$, log L, RMSE | 0.339, 5217.8, 2.16° |

All values are recomputed by `docs/tools/build_figures.py` and checked in JavaScript by
`docs/tools/verify_lab.js`.

## 7.2 What the numbers say

* **The camera measurement is good; the model is the bottleneck.** The 1° quantisation contributes a
  0.29° standard deviation, while the best parametric model leaves 7.1° RMS. The residuals are periodic,
  which is a signature of model error, not of noise.
* **The magnet is not a free linear damped oscillator.** The measured amplitude does not decay to zero
  (≈ 28° → ≈ 10° → ≈ 15°) and the frequency rises (≈ 3.7 → ≈ 4.4 rad/s). Linear drifts of $\beta$ and
  $\omega$ capture the first half of the record only.
* **The Kalman filter compensates for the model.** With a large process noise it tracks the data where
  the model fails (RMSE 2.7°). Its likelihood-based tuning is a principled way to choose $\sigma_q$, but the
  resulting filter mostly smooths the measurements rather than enforcing oscillator physics (chapter 6.4).

## 7.3 Known limitations of the current code

| Area | Limitation | Consequence |
|------|------------|-------------|
| All scripts | Input and output paths are hard-coded absolute Windows paths | they must be edited before running (see the README's Usage section) |
| `aruco_library.detect_ArUco` | `len(ids)` with `ids = None` when no marker is visible | `TypeError` on the first frame without a marker |
| `maglev_rotation_CV.py` | only the first detected id is recorded | several markers in view make the series ambiguous |
| `maglev_rotation_CV.py` | the output video is written at a fixed 20 FPS | a 30 FPS input plays back 1.5× slower |
| `Calculate_orientation` | angles in $[0, 360]$, no unwrapping | a rotation through 0° would create ±360° jumps (does not occur in this dataset) |
| `Calculate_orientation` | in-plane angle only, no camera calibration or pose estimation | camera tilt biases the angle (lab step 2) |
| Fitting | no uncertainty reported (`pcov` is ignored) | parameter confidence intervals are unknown |
| `find_peaks` guesses | no `prominence`/`distance`, offset not removed before the log decrement | spurious peaks; the $\beta$ guess is approximate (still converges here) |
| Advanced model | $\beta(t)t$ and $\omega(t)t$ make the instantaneous rates $\beta_0 + 2\beta_1 t$ and $\omega_0 + 2\omega_1 t$ | the script's $\beta(t)$, $\omega(t)$ plots are not the instantaneous rates |
| `Kalman.py` | $F_k$ has no restoring term; update uses $K[0]$ for both states; $R$ includes model error | see chapter 6.4 |
| Dependencies | legacy `cv2.aruco.detectMarkers` | requires `opencv-contrib-python` 4.7 to 4.x |

## 7.4 Directions that follow from the analysis

These are suggestions, not implemented features:

1. Add the restoring term to the filter, e.g. the Euler discretisation $`F_k = \begin{bmatrix}1 & \Delta t\\ -(\omega^2+\beta^2)\Delta t & 1 - 2\beta\Delta t\end{bmatrix}`$ with the fitted $\beta(t_k)$, $\omega(t_k)$ (since $\omega_n^2 = \omega_d^2 + \beta^2$),
   and the offset to the state, then re-tune $\sigma_q$.
2. Use the full gain vector in the update (the lab shows the likelihood gain).
3. Estimate $R$ from a recording of the marker at rest, and enable sub-pixel corner refinement
   (`CORNER_REFINE_SUBPIX`) to reduce corner noise.
4. Model energy input or a non-linear restoring torque to explain the amplitude floor and the frequency drift.
5. Report $\sqrt{\operatorname{diag}(\texttt{pcov})}$ as parameter standard errors.

[← 6 · Kalman filter](06-kalman-filter.md) · [Guide index](README.md)
