# 3 · Angle extraction: from four corners to one number

[← 2 · ArUco detection](02-aruco-detection.md) · [Guide index](README.md) · Next: [4 · Physical model →](04-physical-model.md)

**Question:** how is the rotation angle computed from the corners, and how accurate is it?
**Code:** `aruco_library.Calculate_orientation()` · **Lab:** [step 2](https://cocopops9.github.io/MaglevCV/#angle)

<p align="center">
  <img src="assets/angle_geometry.svg" alt="Geometry of Calculate_orientation over a full turn" width="100%">
</p>

## 3.1 Geometry

With corners $tl, tr, br, bl$ (image pixels, $y$ pointing down), the code builds two points and **negates
$y$** so that the usual mathematical orientation applies ($+y$ up, angles counter-clockwise):

```math
\mathbf{top} = \left(\tfrac{x_{tl}+x_{tr}}{2},\; -\tfrac{y_{tl}+y_{tr}}{2}\right),\qquad
\mathbf{c} = \left(\tfrac{x_{tl}+x_{tr}+x_{br}+x_{bl}}{4},\; -\tfrac{y_{tl}+y_{tr}+y_{br}+y_{bl}}{4}\right)
```

The vector $\mathbf{d} = \mathbf{top}-\mathbf{c}$ points from the centre to the middle of the marker's own top
edge, i.e. it is the marker's "up" direction. The angle is the direction of $\mathbf d$ measured
counter-clockwise from the image's $+x$ axis (the code comments say "relative to the z-axis"; the quantity
actually computed is this in-plane angle). An upright marker gives 90°.

## 3.2 Extending the arctangent to a full turn

`np.arctan(dy/dx)` only returns values in $(-90°, 90°)$, because $dy/dx$ cannot distinguish $\mathbf d$ from
$-\mathbf d$. The code restores the quadrant from the signs of $dx$ and $dy$:

| Condition | Rule | Output range |
|-----------|------|--------------|
| $dx \ge 0,\ dy \ge 0$ | $\text{round}(\arctan(dy/dx))$ | 0° … 90° |
| $dx < 0$ | $180 + \text{round}(\arctan(dy/dx))$ | 90° … 270° |
| $dx \ge 0,\ dy < 0$ | $360 + \text{round}(\arctan(dy/dx))$ | 270° … 360° |

This is exactly $\operatorname{atan2}(dy, dx)$ mapped to $[0°, 360°]$. Details that follow from the code:

* **Rounding happens before the offset is added**, so the output is always an integer, and equals
  $\text{round}(\operatorname{atan2})$. Python's `round` sends exact halves to the nearest even integer.
* For an angle in $(359.5°, 360°)$ the third branch gives $360 + 0 = 360$, so **360 and 0 can both
  appear** for the same orientation.
* For a perfectly vertical $\mathbf d$, $dx = 0$. The corners are `float32` NumPy values, so the division gives
  $\pm\infty$ with a warning rather than an exception; $\arctan(\pm\infty) = \pm 90°$ and the branches still
  return 90° or 270°. The `except` block is therefore never reached in practice.

## 3.3 What limits the accuracy

| Source | Effect | Size |
|--------|--------|------|
| Integer rounding | uniform error in $[-0.5°, 0.5°]$ | std $= 1/\sqrt{12} \approx 0.29°$ |
| Corner noise (no sub-pixel refinement) | random error, larger for small markers | for corner noise $\sigma_p$ and a marker of side $L$ px, roughly $\sigma_p / (L/2)$ rad; try it in the lab |
| Perspective | systematic error if the marker plane is not parallel to the image | the image angle $a'$ of a plane tilted by $\tau$ satisfies $\tan a' \approx \tan a \cos\tau$ (orthographic approximation) |
| Centre definition | the mean of the four corners is not the projected centre under perspective | small, grows with tilt |

The lab's step 2 implements the same function in JavaScript (verified against the Python version) on a
marker projected by a pinhole camera. Increasing the camera tilt shows the systematic error; increasing the
corner noise shows the spread of 2000 simulated detections.

> In the recorded data the angle stays between 24° and 84°, far from the 0°/360° seam, so no angle
> unwrapping is needed. A marker rotating through 0° would produce jumps of ±360° in the CSV.

## 3.4 Writing the time series

`maglev_rotation_CV.py` reads `frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)` after each `cap.read()`, so the
first frame has `frame_no = 1`, and stores

```math
t_k = \frac{n_\text{frame}}{\text{FPS}}, \qquad \theta_k = \text{angle of the first detected marker (degrees)}
```

as one CSV row `t_k, θ_k`, without a header. The resulting `measured_rotation.csv` has 5926 rows with
$\Delta t = 1/30$ s.

**Hand-off to chapter 4:** a uniformly sampled, 1°-quantised angle signal $\theta_k$ in degrees.

[← 2 · ArUco detection](02-aruco-detection.md) · Next: [4 · Physical model →](04-physical-model.md)
