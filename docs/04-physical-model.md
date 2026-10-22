# 4 · Physical model: the damped rotational oscillator

[← 3 · Angle extraction](03-angle-extraction.md) · [Guide index](README.md) · Next: [5 · Parameter estimation →](05-parameter-estimation.md)

**Question:** which equation should $\theta(t)$ obey, and what do its parameters mean?
**Code:** `ideal_harmonic_oscillator.py` · **Lab:** [step 3](https://cocopops9.github.io/MaglevCV/#physics)

<p align="center">
  <img src="assets/oscillator.svg" alt="Ideal damped rotational oscillator and its phase portrait" width="100%">
</p>

## 4.1 Equation of motion

A rigid body with moment of inertia $I$, held near a rest orientation by a restoring torque proportional to
the deviation (stiffness $k$) and slowed by a torque proportional to the angular velocity (damping $c$), obeys

```math
I\,\ddot\theta + c\,\dot\theta + k\,\theta = 0 .
```

Dividing by $I$ and introducing the state $x = [\theta,\ \dot\theta]^T$ gives a linear time-invariant system,
with the names used in the script ($\alpha = k/I$, $\beta = c/I$):

```math
\dot x = A x,\qquad
A = \begin{bmatrix} 0 & 1 \\ -\alpha & -\beta \end{bmatrix},\qquad
\theta = C x,\quad C = \begin{bmatrix} 1 & 0 \end{bmatrix}.
```

## 4.2 Exact solution with the matrix exponential

For $\dot x = Ax$ the solution is $x(t) = e^{At}x_0$, where $e^{At} = \sum_{n\ge 0} (At)^n / n!$. The script
evaluates it numerically with `scipy.linalg.expm` at 1000 instants in $[0, 10]$ s:

```python
A = np.array([[0, 1], [-alpha, -beta]])     # alpha = 15, beta = 1  (I = 1, k = 15, c = 1)
x0 = np.array([[np.pi / 4], [np.pi]])        # theta0 = 45 deg, omega0 = 180 deg/s
theta[i] = (C @ expm(A * t) @ x0).item()
```

This is exact for any damping regime (no time-stepping error). The lab uses the equivalent closed form for
$2\times2$ matrices: with $s = \operatorname{tr}A/2$ and $B = A - sI$ (for which $B^2 = q^2 I$),

```math
e^{At} = e^{st}\left[\cosh(qt)\,I + \frac{\sinh(qt)}{q}\,B\right],
```

where $\cosh, \sinh$ become $\cos, \sin$ when $q^2 < 0$ (verified against `expm` to $10^{-9}$).

## 4.3 Regimes and the form used for fitting

The eigenvalues of $A$ are $\lambda = -\frac{\beta}{2} \pm \sqrt{\frac{\beta^2}{4} - \alpha}$.

| Regime | Condition | Behaviour |
|--------|-----------|-----------|
| Under-damped | $c^2 < 4kI$ (damping ratio $\zeta = \frac{c}{2\sqrt{kI}} < 1$) | decaying oscillation |
| Critically damped | $\zeta = 1$ | fastest return without oscillation |
| Over-damped | $\zeta > 1$ | slow return without oscillation |

In the under-damped case the solution can be rewritten as

```math
\theta(t) = A_0\, e^{-\frac{c}{2I} t} \cos(\omega_d t + \varphi),\qquad
\omega_d = \sqrt{\frac{k}{I} - \left(\frac{c}{2I}\right)^2},
```

```math
A_0 = \sqrt{\theta_0^2 + \left(\frac{\omega_0 + \frac{c}{2I}\theta_0}{\omega_d}\right)^2},\qquad
\varphi = \operatorname{atan2}\!\left(-\frac{\omega_0 + \frac{c}{2I}\theta_0}{\omega_d},\ \theta_0\right).
```

This is **the model every fitting script uses**, `A * exp(-beta*t) * cos(omega*t + phi)`. Note the two
meanings of $\beta$:

| Name | In `ideal_harmonic_oscillator.py` | In the fitting scripts |
|------|-----------------------------------|------------------------|
| $\beta$ | $c/I$ (coefficient of $\dot\theta$) | $c/2I$ (decay rate of the envelope) |

For the script's values ($I=1$, $k=15$, $c=1$, $\theta_0=\pi/4$, $\omega_0=\pi$): $\zeta = 0.129$,
$\beta_\text{fit} = 0.5\ \text{s}^{-1}$, $\omega_d = \sqrt{14.75} = 3.8406$ rad/s, $A_0 = 1.2098$ rad,
$\varphi = -49.52°$. Fitting the simulated curve with `model_estimation_naive.py` returns exactly these values,
which validates the fitting code before it meets real data (chapter 5).

The phase portrait ($\theta$ against $\dot\theta$) of an under-damped oscillator is a spiral into the origin:
each turn is one period, and the ratio between successive turns is $e^{-\beta_\text{fit} T}$, the logarithmic
decrement used in chapter 5 to guess $\beta$.

## 4.4 Extensions needed for the real magnet

Two terms are added when the model meets the measured data:

* **Offset.** The camera measures the angle from the image $x$ axis, not from the magnet's rest orientation.
  A constant `offset` (≈ 0.87 rad ≈ 50°) absorbs this.
* **Slowly varying parameters** (`model_estimanion_advanced.py`):
  $\beta(t) = \beta_0 + \beta_1 t$ and $\omega(t) = \omega_0 + \omega_1 t$, inserted as
  $A e^{-\beta(t)\,t}\cos(\omega(t)\,t + \varphi) + \text{offset}$.
  Because the products are $\beta(t)\,t$ and $\omega(t)\,t$, the **instantaneous** decay rate and frequency are
  $\frac{d}{dt}[\beta(t)t] = \beta_0 + 2\beta_1 t$ and $\frac{d}{dt}[\omega(t)t] = \omega_0 + 2\omega_1 t$,
  not $\beta(t)$ and $\omega(t)$ as plotted by the script.

What the linear model cannot represent: a homogeneous linear oscillator can only lose amplitude
exponentially. The measured amplitude instead decreases from about 28° to about 10° and then **rises again**
to about 15° (chapter 5, figure "envelope"), and its oscillation frequency rises from about 3.7 to about
4.4 rad/s. Both point to energy input and/or non-linear restoring torque, which are outside the model; the
Kalman filter of chapter 6 is how the project copes with this mismatch.

**Hand-off to chapter 5:** a parametric family $\theta(t; p)$ (naive: 4 parameters, simple: 5, advanced: 7)
whose parameters must be estimated from the data.

[← 3 · Angle extraction](03-angle-extraction.md) · Next: [5 · Parameter estimation →](05-parameter-estimation.md)
