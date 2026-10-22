# 5 · Parameter estimation: fitting the oscillator to the data

[← 4 · Physical model](04-physical-model.md) · [Guide index](README.md) · Next: [6 · Kalman filter →](06-kalman-filter.md)

**Question:** which parameters make the model closest to the measurements, and how are they found?
**Code:** `model_estimation_naive.py`, `model_estimation_naive_4real_data.py`, `model_estimanion_simple.py`,
`model_estimanion_advanced.py` · **Lab:** [step 4](https://cocopops9.github.io/MaglevCV/#fitting)

<p align="center">
  <img src="assets/fitting.svg" alt="Naive, simple and advanced fits on the measured data" width="100%">
</p>

## 5.1 Non-linear least squares

All four scripts call `scipy.optimize.curve_fit(model, t, θ, ...)` with $\theta$ in radians. Without bounds,
`curve_fit` uses the **Levenberg–Marquardt** algorithm (MINPACK) to solve

```math
\hat p = \arg\min_p \; \sum_{k} \big(\theta_k - f(t_k; p)\big)^2 .
```

Each iteration linearises the model around the current $p$ (Jacobian $J_{kj} = \partial f(t_k)/\partial p_j$)
and solves

```math
\left(J^TJ + \lambda\,\operatorname{diag}(J^TJ)\right)\delta = J^T r, \qquad r_k = \theta_k - f(t_k; p).
```

A small $\lambda$ gives a Gauss–Newton step (fast near the minimum), a large $\lambda$ a short gradient-descent
step (safe far from it); $\lambda$ is decreased after a successful step and increased after a failed one.
The method converges to a **local** minimum. Oscillator models are strongly non-convex in $\omega$ and
$\varphi$ (a wrong frequency lines up peaks with troughs), so the starting point $p_0$ decides the result.
The scripts report the fit quality as the sum of squared residuals in degrees,
$\text{SSR} = \sum_k (\theta_k^{\circ} - f^{\circ}(t_k))^2$.

## 5.2 Three models, three starting points

| Script | Model | $p_0$ | Result on `measured_rotation.csv` |
|--------|-------|-------|-----------------------------------|
| `model_estimation_naive_4real_data.py` | $A e^{-\beta t}\cos(\omega t + \varphi)$ | `curve_fit` default: all ones | SSR = 1,084,427 deg², RMS 13.5° |
| `model_estimanion_simple.py` | $A e^{-\beta t}\cos(\omega t + \varphi) + o$ | `compute_initial_guess` (peaks) | SSR = 715,619 deg², RMS 11.0° |
| `model_estimanion_advanced.py` | $A e^{-(\beta_0+\beta_1 t)t}\cos((\omega_0+\omega_1 t)t + \varphi) + o$ | peaks + drift estimates | SSR = 300,861 deg², RMS 7.1° |

(RMS = $\sqrt{\text{SSR}/5926}$. The numbers are reproduced by `docs/tools/build_figures.py`; the stored
parameter files match a fresh run to about 8 significant digits.)

### Naive: works on ideal data, fails on real data

`model_estimation_naive.py` fits the simulated curve of chapter 4, starting from all ones, and recovers
$A = 1.2098$, $\beta = 0.5000$, $\omega = 3.8406$, $\varphi = -49.52°$: the exact values. On the real data the
same procedure fails for two reasons: the model has **no offset**, while the data oscillate around ≈ 50°;
and $p_0$ is arbitrary. SciPy stops at $\omega \approx -1.4\times10^{-6}$ rad/s, i.e. a curve with no
oscillation at all. (The lab's own Levenberg–Marquardt, started from the same ones, can stop in a different
but equally poor local minimum: the failure is structural, not an implementation detail.)

### Simple: offset plus a data-driven initial guess

`compute_initial_guess` derives $p_0$ from the signal:

| Parameter | Estimate | Value on the real data |
|-----------|----------|------------------------|
| peaks | `scipy.signal.find_peaks(theta)` with no options (every strict local maximum, plateaus reduced to their middle) | 246 peaks |
| $\beta$ | logarithmic decrement $\delta = \ln\lvert\theta_{p_1}/\theta_{p_2}\rvert$, $\beta = \delta/T$ | 0 (the first two peaks are both 82°) |
| $\omega$ | $2\pi/T$, $T = t_{p_2} - t_{p_1}$ | 3.6249 rad/s |
| offset | mean of $\theta$ | 0.8698 rad (49.8°) |
| $A$ | $(\max\theta - \min\theta)/2$ | 0.5236 rad (30°) |
| $\varphi$ | $\arccos((\theta_0 - o)/A)$ | 0.4383 rad |

Two remarks follow from the code. The log decrement is computed on the raw peak values, which include the
≈ 50° offset, so it is not the decrement of the oscillation itself. And because the data are quantised and noisy,
`find_peaks` without a `prominence` or `distance` also returns small local maxima on the slopes and in the
troughs (for example at 9.1 s, value 29°); only the first two and the last two peaks are used, so the guess
is still usable. From this $p_0$ Levenberg–Marquardt converges to
$A = 0.682$ rad, $\beta = 0.0297$ s⁻¹, $\omega = 3.732$ rad/s, $\varphi = 0.130$, offset $= 0.870$ rad
(`parameters_constant.csv`).

### Advanced: letting the parameters drift

The simple model forces one decay rate and one frequency over 197 s. The advanced model lets both vary
linearly. The extra initial guesses are

```math
\beta_1^{(0)} = \frac{2(\bar\beta - \beta_0^{(0)})}{T_\text{tot}},\quad
\bar\beta = \frac{\ln\lvert\theta_{p_\text{first}}/\theta_{p_\text{last}}\rvert}{T_\text{tot}},\qquad
\omega_1^{(0)} = \frac{2\pi/T_\text{end} - \omega_0^{(0)}}{T_\text{tot}},
```

($1.46\times10^{-5}$ s⁻² and $4.96\times10^{-3}$ rad/s² here). The optimum
(`parameters_variable.csv`) is $A = 0.394$, $\beta_0 = -0.0113$, $\beta_1 = 1.83\times10^{-4}$,
$\omega_0 = 3.541$, $\omega_1 = 4.12\times10^{-3}$, $\varphi = 1.534$, offset $= 0.870$. It lowers the SSR by
58 % with respect to the simple model. Its instantaneous behaviour (chapter 4.4):

| $t$ (s) | decay rate $\beta_0 + 2\beta_1 t$ (s⁻¹) | frequency $\omega_0 + 2\omega_1 t$ (rad/s) | model amplitude | measured frequency |
|---------|------------------------------------------|---------------------------------------------|-----------------|--------------------|
| 0 | −0.011 (growing) | 3.54 | 22.5° | 3.68 (0–30 s) |
| 30 | ≈ 0 | 3.79 | 26.8° | |
| 100 | 0.025 | 4.37 | 11.2° | 4.32 (80–110 s) |
| 197.5 | 0.061 | 5.17 | 0.17° | 4.44 (160–197 s) |

(Measured frequencies: peak of the windowed FFT of the data.) The fit follows the data well for the first
half. Late in the record its amplitude is almost zero, so a wrong phase costs almost nothing in the SSR,
and its frequency overshoots the measured one.

<p align="center">
  <img src="assets/envelope.svg" alt="Measured amplitude compared with the two model envelopes" width="100%">
</p>

## 5.3 Reading the residuals

All scripts plot the residuals $\theta_k - f(t_k;\hat p)$. For a correct model with independent measurement
noise they would look like white noise of a few tenths of a degree (the 1° quantisation alone gives 0.29°).
Here they oscillate at the signal frequency with amplitudes of roughly 10° to 25°: the error is dominated by **model
mismatch**, not by sensor noise. This is the motivation for chapter 6.

**Hand-off to chapter 6:** `parameters_variable.csv` ($\beta_0, \beta_1$ are used by the filter's transition
matrix) and the residuals of that model (their standard deviation becomes $\sigma_z$).

[← 4 · Physical model](04-physical-model.md) · Next: [6 · Kalman filter →](06-kalman-filter.md)
