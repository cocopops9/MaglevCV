"""
Regenerates every animated SVG figure in docs/assets and the data bundle used by the
interactive lab (docs/lab/data.js).

Every figure is computed from the repository itself:
  * the measured dataset (measured_rotation.csv),
  * the parameter files (parameters_constant.csv, parameters_variable.csv),
  * the ArUco functions in aruco_library.py (imported directly),
  * the same model equations, initial-guess logic and Kalman loop used by the scripts
    (re-stated here because the scripts run their whole pipeline at import time).

Usage (from the repository root):
    pip install -r requirements.txt
    python docs/tools/build_figures.py
"""

import base64
import csv
import json
import math
import os
import sys

import cv2
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS = os.path.join(ROOT, "docs", "assets")
LAB = os.path.join(ROOT, "docs", "lab")
sys.path.insert(0, ROOT)

import aruco_library  # noqa: E402  (imported after the path is set)

os.makedirs(ASSETS, exist_ok=True)
os.makedirs(LAB, exist_ok=True)


# ----------------------------------------------------------------------------------------
# Palette and SVG helpers
# ----------------------------------------------------------------------------------------

BG_TOP = "#0b1020"
BG_BOTTOM = "#121a33"
GRID = "#24304f"
AXIS = "#3a4770"
TEXT = "#e6edf7"
MUTED = "#8d9ab8"
CYAN = "#38bdf8"      # measurements
ORANGE = "#f59e0b"    # constant-parameter model
VIOLET = "#a78bfa"    # variable-parameter model
RED = "#f87171"       # naive model
GREEN = "#34d399"     # Kalman estimates
PINK = "#f472b6"      # innovations

SANS = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace"


def svg_open(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
        f"<title>{title}</title>",
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{BG_TOP}"/><stop offset="1" stop-color="{BG_BOTTOM}"/>'
        "</linearGradient>",
        '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="2.2" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="18" fill="url(#bg)"/>',
    ]


def text(x, y, s, size=14, fill=TEXT, anchor="start", weight="400", family=SANS, extra=""):
    s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')


def fmt_values(values, digits=1):
    return ";".join(f"{v:.{digits}f}" for v in values)


def save(name, parts):
    parts.append("</svg>")
    path = os.path.join(ASSETS, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"wrote {os.path.relpath(path, ROOT)}  ({os.path.getsize(path) / 1024:.1f} kB)")


class Plot:
    """Maps data coordinates into a rectangle of the SVG canvas."""

    def __init__(self, x0, y0, w, h, xmin, xmax, ymin, ymax):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax

    def X(self, x):
        return self.x0 + (np.asarray(x) - self.xmin) / (self.xmax - self.xmin) * self.w

    def Y(self, y):
        return self.y0 + self.h - (np.asarray(y) - self.ymin) / (self.ymax - self.ymin) * self.h

    def path(self, xs, ys):
        px, py = self.X(xs), self.Y(ys)
        head = f"M{px[0]:.1f},{py[0]:.1f}"
        return head + "".join(f"L{a:.1f},{b:.1f}" for a, b in zip(px[1:], py[1:]))

    def frame(self, xticks, yticks, xlabel="", ylabel="", xfmt="{:g}", yfmt="{:g}"):
        out = [f'<rect x="{self.x0}" y="{self.y0}" width="{self.w}" height="{self.h}" '
               f'fill="#0e1530" stroke="{AXIS}" rx="6"/>']
        for v in xticks:
            x = self.X(v)
            out.append(f'<line x1="{x:.1f}" y1="{self.y0}" x2="{x:.1f}" y2="{self.y0 + self.h}" '
                       f'stroke="{GRID}" stroke-width="1"/>')
            out.append(text(x, self.y0 + self.h + 16, xfmt.format(v), 11, MUTED, "middle", family=MONO))
        for v in yticks:
            y = self.Y(v)
            out.append(f'<line x1="{self.x0}" y1="{y:.1f}" x2="{self.x0 + self.w}" y2="{y:.1f}" '
                       f'stroke="{GRID}" stroke-width="1"/>')
            out.append(text(self.x0 - 8, y + 4, yfmt.format(v), 11, MUTED, "end", family=MONO))
        if xlabel:
            out.append(text(self.x0 + self.w / 2, self.y0 + self.h + 34, xlabel, 12, MUTED, "middle"))
        if ylabel:
            cx, cy = self.x0 - 44, self.y0 + self.h / 2
            out.append(text(cx, cy, ylabel, 12, MUTED, "middle",
                            extra=f'transform="rotate(-90 {cx} {cy})"'))
        return out


def png_data_uri(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def jpg_data_uri(img, quality=82):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def decimate(xs, ys, columns):
    """Keeps the min and max of each pixel column: visually lossless for dense signals."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    edges = np.linspace(xs[0], xs[-1], columns + 1)
    bins = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, columns - 1)
    keep = []
    for b in range(columns):
        idx = np.nonzero(bins == b)[0]
        if len(idx) == 0:
            continue
        lo, hi = idx[np.argmin(ys[idx])], idx[np.argmax(ys[idx])]
        keep.extend(sorted({int(lo), int(hi)}))
    keep = np.array(keep)
    return xs[keep], ys[keep]


# ----------------------------------------------------------------------------------------
# Data and models (identical equations to the scripts)
# ----------------------------------------------------------------------------------------

def read_measurements(filename):
    time, theta = [], []
    with open(filename, "r") as file:
        for row in csv.reader(file):
            time.append(float(row[0]))
            theta.append(float(row[1]))
    return np.array(time), np.array(theta)


def read_parameters(filename):
    with open(filename, "r") as file:
        return {row[0]: float(row[1]) for row in csv.reader(file)}


def damped_oscillator_naive(t, A, beta, omega, phi):
    return A * np.exp(-beta * t) * np.cos(omega * t + phi)


def damped_oscillator(t, A, beta, omega, phi, offset):
    return A * np.exp(-beta * t) * np.cos(omega * t + phi) + offset


def damped_oscillator_variable(t, A, beta0, beta1, omega0, omega1, phi, offset):
    beta = beta0 + beta1 * t
    omega = omega0 + omega1 * t
    return A * np.exp(-beta * t) * np.cos(omega * t + phi) + offset


def kalman_run(time, z, params, sigma_q, sigma_z):
    """Same loop as Kalman.py, including its update x_est = x_pred + K[0,0] * y."""
    dt = time[1] - time[0]
    R = sigma_z ** 2
    Q = np.array([[0, 0], [0, sigma_q ** 2]])
    x_est = np.array([z[0], (z[1] - z[0]) / dt])
    P_est = np.eye(2)
    H = np.array([[1, 0]])
    n = len(time)
    log = {k: np.zeros(n) for k in ("x_pred", "P_pred", "x_est", "P_est", "innov", "S")}
    log_likelihood = 0.0
    for k in range(n):
        beta_tk = params["beta0"] + params["beta1"] * time[k]
        F_k = np.array([[1, dt], [0, 1 - beta_tk * dt]])
        x_pred = F_k @ x_est
        P_pred = F_k @ P_est @ F_k.T + Q
        y_k = z[k] - (H @ x_pred)[0]
        S_k = (H @ P_pred @ H.T)[0, 0] + R
        K_k = (P_pred @ H.T) / S_k
        x_est = x_pred + K_k[0, 0] * y_k
        P_est = (np.eye(2) - K_k @ H) @ P_pred
        log["x_pred"][k], log["P_pred"][k] = x_pred[0], P_pred[0, 0]
        log["x_est"][k], log["P_est"][k] = x_est[0], P_est[0, 0]
        log["innov"][k], log["S"][k] = y_k, S_k
        log_likelihood += -0.5 * (np.log(2 * np.pi * S_k) + y_k ** 2 / S_k)
    return log_likelihood, log


time, theta_deg = read_measurements(os.path.join(ROOT, "measured_rotation.csv"))
theta = np.deg2rad(theta_deg)
P_CONST = read_parameters(os.path.join(ROOT, "parameters_constant.csv"))
P_VAR = read_parameters(os.path.join(ROOT, "parameters_variable.csv"))

model_const_deg = np.rad2deg(damped_oscillator(time, *P_CONST.values()))
model_var_deg = np.rad2deg(damped_oscillator_variable(time, *P_VAR.values()))

# model_estimation_naive_4real_data.py: no offset, no initial guess (p0 = all ones)
popt_naive, _ = curve_fit(damped_oscillator_naive, time, theta, maxfev=200000)
model_naive_deg = np.rad2deg(damped_oscillator_naive(time, *popt_naive))

SSR = {
    "naive": float(np.sum((theta_deg - model_naive_deg) ** 2)),
    "constant": float(np.sum((theta_deg - model_const_deg) ** 2)),
    "variable": float(np.sum((theta_deg - model_var_deg) ** 2)),
}

residuals_var = theta - np.deg2rad(model_var_deg)
SIGMA_Z = float(np.std(residuals_var))
SIGMA_Q_GRID = np.logspace(-7, 1, num=50)
LIKELIHOOD = np.array([kalman_run(time, theta, P_VAR, s, SIGMA_Z)[0] for s in SIGMA_Q_GRID])
BEST = int(np.argmax(LIKELIHOOD))
SIGMA_Q_BEST = float(SIGMA_Q_GRID[BEST])
_, KF = kalman_run(time, theta, P_VAR, SIGMA_Q_BEST, SIGMA_Z)
RMSE_KF = float(np.sqrt(np.mean((KF["x_est"] - theta) ** 2)))
SNR_DB = float(10 * np.log10(np.mean(theta ** 2) / SIGMA_Z ** 2))

print("SSR", SSR)
print("naive popt", popt_naive)
print(f"sigma_z = {SIGMA_Z:.4f} rad, sigma_q* = {SIGMA_Q_BEST:.4f}, logL* = {LIKELIHOOD[BEST]:.2f}, "
      f"RMSE = {np.degrees(RMSE_KF):.3f} deg, SNR = {SNR_DB:.2f} dB")

# DICT_5X5_50, id 0: 7x7 cells including the black border (1 = white)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
MARKER_BITS = (cv2.aruco.generateImageMarker(ARUCO_DICT, 0, 7) // 255).astype(int)


def marker_cells(cx, cy, size):
    """Rects of the white cells of the marker, centred at (cx, cy), side 'size'."""
    cell = size / 7
    x0, y0 = cx - size / 2, cy - size / 2
    out = [f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{size:.1f}" height="{size:.1f}" fill="#111"/>']
    # One path for all white cells, so adjacent cells do not show anti-aliasing seams
    d = "".join(f"M{x0 + c * cell:.2f},{y0 + r * cell:.2f}h{cell:.2f}v{cell:.2f}h{-cell:.2f}z"
                for r in range(7) for c in range(7) if MARKER_BITS[r, c])
    out.append(f'<path d="{d}" fill="#f5f5f5"/>')
    return out


def bgr(b, g, r):
    return f"rgb({r},{g},{b})"


# Colours used by aruco_library.mark_ArUco (OpenCV uses BGR order)
MARK_LINE = bgr(105, 24, 0)
MARK_TL = bgr(100, 100, 100)
MARK_TR = bgr(0, 255, 0)
MARK_BR = bgr(100, 100, 255)
MARK_BL = bgr(255, 255, 255)
MARK_CENTRE = bgr(0, 0, 255)
MARK_ID = bgr(0, 0, 200)
MARK_ANGLE = bgr(0, 0, 80)


def tape(values_sorted, x, y, line_h, size, fill, weight="700", family=MONO, anchor="start"):
    """A vertical strip of numbers; translating it shows one value inside a clip window."""
    out = []
    for i, v in enumerate(values_sorted):
        out.append(text(x, y + i * line_h, str(v), size, fill, anchor, weight, family))
    return out


# ----------------------------------------------------------------------------------------
# 1. Hero: the measured rotation replayed on a marker, synchronised with the time series
# ----------------------------------------------------------------------------------------

def build_hero():
    W, H = 960, 430
    p = svg_open(W, H, "MaglevCV: measured rotation replayed")
    seconds, speed = 30.0, 3.0
    m = time <= seconds
    t_win, a_win = time[m], theta_deg[m].astype(int)
    dur = seconds / speed

    p.append(text(32, 46, "MaglevCV", 30, TEXT, weight="800"))
    p.append(text(200, 46, "computer vision · physical modelling · Kalman filtering", 15, MUTED))
    p.append(text(32, 70, f"Replay of the first {seconds:.0f} s of measured_rotation.csv at {speed:.0f}x speed: "
                          "every frame below is a real sample.", 12.5, MUTED))

    # Camera frame panel (light background so the exact OpenCV overlay colours stay visible)
    cx, cy, size = 205, 255, 150
    p.append(f'<rect x="32" y="92" width="346" height="310" rx="12" fill="#cfd5de"/>')
    p.append(text(46, 114, "camera frame + mark_ArUco overlay", 12, "#334155", weight="600"))
    p.append(f'<rect x="32" y="92" width="346" height="310" rx="12" fill="none" stroke="#64748b"/>')

    rot_values = [f"{90 - a} {cx} {cy}" for a in a_win]
    rot_anim = (f'<animateTransform attributeName="transform" type="rotate" calcMode="discrete" '
                f'dur="{dur}s" repeatCount="indefinite" values="{";".join(rot_values)}"/>')
    counter_values = [f"{a - 90}" for a in a_win]

    half = size / 2
    p.append(f'<g transform="rotate(0 {cx} {cy})">{rot_anim}')
    p.extend(marker_cells(cx, cy, size))
    top = (cx, cy - half)
    p.append(f'<line x1="{top[0]}" y1="{top[1]}" x2="{cx}" y2="{cy}" stroke="{MARK_LINE}" stroke-width="3"/>')
    for (px, py), col in [((cx - half, cy - half), MARK_TL), ((cx + half, cy - half), MARK_TR),
                          ((cx + half, cy + half), MARK_BR), ((cx - half, cy + half), MARK_BL)]:
        p.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{col}" stroke="#0008" stroke-width="0.6"/>')
    # Angle label written at the top-edge midpoint, upright as cv2.putText draws it
    p.append(f'<g transform="translate({top[0]} {top[1]})"><g>'
             f'<animateTransform attributeName="transform" type="rotate" calcMode="discrete" '
             f'dur="{dur}s" repeatCount="indefinite" values="{";".join(counter_values)}"/>')
    p.append('<clipPath id="heroTapeClip"><rect x="-4" y="-22" width="44" height="28"/></clipPath>')
    p.append('<g clip-path="url(#heroTapeClip)"><g>')
    lo = int(a_win.min())
    shifts = [f"0 {-(a - lo) * 28}" for a in a_win]
    p.append(f'<animateTransform attributeName="transform" type="translate" calcMode="discrete" '
             f'dur="{dur}s" repeatCount="indefinite" values="{";".join(shifts)}"/>')
    p.extend(tape(list(range(lo, int(a_win.max()) + 1)), 0, 0, 28, 20, MARK_ANGLE, family=SANS))
    p.append("</g></g></g></g>")
    p.append("</g>")
    p.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{MARK_CENTRE}"/>')
    p.append(text(cx + 2, cy, "0", 22, MARK_ID, weight="700"))

    # Time series panel
    pl = Plot(470, 110, 450, 250, 0, seconds, 15, 95)
    p.extend(pl.frame(range(0, 31, 5), range(20, 91, 20), "time (s)", "angle (deg)"))
    p.append(f'<clipPath id="heroReveal"><rect x="{pl.x0}" y="{pl.y0 - 5}" width="0" height="{pl.h + 10}">'
             f'<animate attributeName="width" values="0;{pl.w}" dur="{dur}s" repeatCount="indefinite"/>'
             f'</rect></clipPath>')
    p.append(f'<path d="{pl.path(t_win, a_win)}" fill="none" stroke="{CYAN}" stroke-width="1.6" '
             f'clip-path="url(#heroReveal)" filter="url(#glow)"/>')
    ys = pl.Y(a_win)
    p.append(f'<circle r="5" fill="{TEXT}" cx="{pl.x0}" cy="{ys[0]:.1f}">'
             f'<animate attributeName="cx" values="{pl.X(t_win[0]):.1f};{pl.X(t_win[-1]):.1f}" dur="{dur}s" repeatCount="indefinite"/>'
             f'<animate attributeName="cy" calcMode="discrete" values="{fmt_values(ys)}" dur="{dur}s" repeatCount="indefinite"/>'
             f'</circle>')
    p.append(text(pl.x0, 100, "θ(t) written to measured_rotation.csv", 12.5, TEXT, weight="600"))
    p.append(text(pl.x0 + pl.w, 400, f"{len(time)} samples · 30 fps · 1° resolution", 11.5, MUTED, "end"))
    save("hero.svg", p)


# ----------------------------------------------------------------------------------------
# 2. Pipeline overview
# ----------------------------------------------------------------------------------------

def build_pipeline():
    W, H = 960, 340
    p = svg_open(W, H, "MaglevCV pipeline")
    p.append(text(32, 40, "Pipeline: from pixels to a filtered state estimate", 18, TEXT, weight="700"))

    stages = [
        ("1", "Video", "maglev1.mp4", "OpenCV VideoCapture", "#1e3a5f", CYAN),
        ("2", "ArUco tracking", "aruco_library.py", "maglev_rotation_CV.py", "#1e3a5f", CYAN),
        ("3", "Time series", "measured_rotation.csv", "(t [s], θ [deg])", "#2a2f55", TEXT),
        ("4", "Model fitting", "model_estimation_*.py", "curve_fit (LM)", "#3b2f1a", ORANGE),
        ("5", "Kalman filter", "Kalman.py", "σq by max likelihood", "#123b33", GREEN),
    ]
    bw, bh, gap, y = 164, 104, 19, 92
    x = 32
    centres = []
    for i, (num, title, file_, sub, fill, accent) in enumerate(stages):
        p.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="12" fill="{fill}" stroke="{accent}" stroke-opacity="0.65">'
                 f'<animate attributeName="stroke-opacity" values="0.35;1;0.35" dur="5s" begin="{i}s" repeatCount="indefinite"/></rect>')
        p.append(f'<circle cx="{x + 20}" cy="{y + 22}" r="11" fill="{accent}"/>')
        p.append(text(x + 20, y + 26.5, num, 12, BG_TOP, "middle", "800"))
        p.append(text(x + 38, y + 27, title, 14.5, TEXT, weight="700"))
        p.append(text(x + 12, y + 60, file_, 10.5, accent, family=MONO))
        p.append(text(x + 14, y + 82, sub, 11.5, MUTED))
        centres.append((x, x + bw))
        x += bw + gap
    for i in range(len(stages) - 1):
        x1, x2 = centres[i][1], centres[i + 1][0]
        yy = y + bh / 2
        p.append(f'<line x1="{x1 + 3}" y1="{yy}" x2="{x2 - 6}" y2="{yy}" stroke="{MUTED}" stroke-width="2" '
                 f'stroke-dasharray="4 5"><animate attributeName="stroke-dashoffset" values="18;0" dur="0.8s" repeatCount="indefinite"/></line>')
        p.append(f'<path d="M{x2 - 9},{yy - 5} L{x2 - 2},{yy} L{x2 - 9},{yy + 5}" fill="none" stroke="{MUTED}" stroke-width="2"/>')
        p.append(f'<circle r="4" cx="{x1}" cy="{yy}" fill="{TEXT}" opacity="0" filter="url(#glow)">'
                 f'<animate attributeName="cx" values="{x1};{x2}" dur="5s" begin="{i + 0.5}s" keyTimes="0;0.2;1" '
                 f'keySplines="0.4 0 0.2 1;0 0 1 1" calcMode="spline" repeatCount="indefinite"/>'
                 f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.2;0.21;1" dur="5s" begin="{i + 0.5}s" repeatCount="indefinite"/>'
                 f'</circle>')

    # Outputs and the validation branch
    yb = 240
    p.append(f'<rect x="32" y="{yb}" width="347" height="76" rx="12" fill="#1a1f3a" stroke="{VIOLET}" stroke-opacity="0.6"/>')
    p.append(text(46, yb + 24, "Validation branch", 13, VIOLET, weight="700"))
    p.append(text(46, yb + 46, "ideal_harmonic_oscillator.py →", 10.5, MUTED, family=MONO))
    p.append(text(46, yb + 62, "model_estimation_naive.py", 10.5, MUTED, family=MONO))
    p.append(f'<rect x="401" y="{yb}" width="347" height="76" rx="12" fill="#1a1f3a" stroke="{ORANGE}" stroke-opacity="0.6"/>')
    p.append(text(415, yb + 24, "Fitted parameters", 13, ORANGE, weight="700"))
    p.append(text(415, yb + 46, "parameters_constant.csv", 10.5, MUTED, family=MONO))
    p.append(text(415, yb + 62, "parameters_variable.csv", 10.5, MUTED, family=MONO))
    p.append(f'<rect x="770" y="{yb}" width="158" height="76" rx="12" fill="#13302b" stroke="{GREEN}" stroke-opacity="0.6"/>')
    p.append(text(784, yb + 24, "Outputs", 13, GREEN, weight="700"))
    p.append(text(784, yb + 46, "estimates θ(t),", 11.5, MUTED))
    p.append(text(784, yb + 62, "innovations, RMSE", 11.5, MUTED))
    # connectors from stage 4 to parameters, parameters to stage 5
    s4 = centres[3]
    p.append(f'<path d="M{(s4[0] + s4[1]) / 2},{y + bh} L{(s4[0] + s4[1]) / 2},{yb}" stroke="{ORANGE}" stroke-width="1.6" stroke-dasharray="4 4" fill="none"/>')
    s5 = centres[4]
    p.append(f'<path d="M{(s5[0] + s5[1]) / 2},{y + bh} L{(s5[0] + s5[1]) / 2},{yb}" stroke="{GREEN}" stroke-width="1.6" stroke-dasharray="4 4" fill="none"/>')
    save("pipeline.svg", p)


# ----------------------------------------------------------------------------------------
# 3. ArUco detection, step by step, on a synthetic frame processed by OpenCV
# ----------------------------------------------------------------------------------------

def build_detection():
    # Synthetic camera frame: the marker in perspective, with blur, a gradient and noise
    rng = np.random.default_rng(7)
    S = 240
    marker = cv2.aruco.generateImageMarker(ARUCO_DICT, 0, 140)
    quiet = cv2.copyMakeBorder(marker, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    src = np.float32([[0, 0], [180, 0], [180, 180], [0, 180]])
    dst = np.float32([[58, 40], [196, 62], [182, 205], [40, 178]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    gradient = np.tile(np.linspace(150, 210, S, dtype=np.float32), (S, 1))
    frame = cv2.warpPerspective(quiet.astype(np.float32), Hm, (S, S), borderValue=-1)
    frame = np.where(frame < 0, gradient, frame * 0.82 + 20)
    frame = cv2.GaussianBlur(frame, (3, 3), 0.8) + rng.normal(0, 6, (S, S))
    gray = np.clip(frame, 0, 255).astype(np.uint8)
    color = cv2.merge([np.clip(gray * 0.92 + 8, 0, 255).astype(np.uint8), gray,
                       np.clip(gray * 1.05, 0, 255).astype(np.uint8)])

    # Default DetectorParameters: windows 3..23 step 10, constant C = 7. Window 13 shown.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 7)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_img = cv2.cvtColor(thresh // 3, cv2.COLOR_GRAY2BGR)
    quads = []
    for c in contours:
        if len(c) < 0.03 * 4 * S:  # minMarkerPerimeterRate
            continue
        approx = cv2.approxPolyDP(c, 0.03 * len(c), True)  # polygonalApproxAccuracyRate
        cv2.drawContours(contour_img, [c], -1, (120, 120, 120), 1)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx)
    for q in quads:
        cv2.polylines(contour_img, [q], True, (80, 220, 255), 2)

    detected = aruco_library.detect_ArUco(color.copy())
    angles = aruco_library.Calculate_orientation(detected)
    corners = detected[0][0].astype(np.float32)

    # Perspective removal: 4 px per cell (perspectiveRemovePixelPerCell), Otsu, cell voting
    cells = 7
    ppc = 4
    side = cells * ppc
    Hc = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]]))
    canon = cv2.warpPerspective(gray, Hc, (side, side), flags=cv2.INTER_NEAREST)
    _, canon_bin = cv2.threshold(canon, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    big = cv2.resize(canon, (S, S), interpolation=cv2.INTER_NEAREST)
    bits = np.zeros((cells, cells), int)
    margin = int(0.13 * ppc)
    for r in range(cells):
        for c in range(cells):
            cell = canon_bin[r * ppc + margin:(r + 1) * ppc - margin, c * ppc + margin:(c + 1) * ppc - margin]
            bits[r, c] = int(cell.mean() > 127)
    assert (bits == MARKER_BITS).all(), "cell voting did not recover the dictionary bits"

    output = aruco_library.mark_ArUco(color.copy(), detected, angles)

    steps = [
        ("1 · Input frame", "BGR image read by cv2.VideoCapture", jpg_data_uri(color)),
        ("2 · Grayscale", "cv2.cvtColor(img, COLOR_BGR2GRAY)", jpg_data_uri(gray)),
        ("3 · Adaptive threshold", "mean of a 13×13 window minus C = 7, inverted", png_data_uri(thresh)),
        ("4 · Contours → quads", f"approxPolyDP keeps convex 4-gons ({len(quads)} found)", jpg_data_uri(contour_img, 88)),
        ("5 · Perspective removal", "warp to 7×7 cells (4 px/cell), Otsu, cell vote", png_data_uri(canon)),
        ("6 · Bits → identifier", "inner 5×5 bits matched in 4 rotations → ID 0", None),
        ("7 · Overlay (mark_ArUco)", f"corners, centre, id, angle = {angles[0]}°", jpg_data_uri(output, 88)),
    ]

    W, H = 960, 380
    p = svg_open(W, H, "ArUco detection step by step")
    p.append(text(32, 40, "How cv2.aruco.detectMarkers finds the marker (real OpenCV outputs on a synthetic frame)", 16, TEXT, weight="700"))
    step_dur = 2.6
    total = step_dur * len(steps)
    ix, iy, isz = 60, 80, 260
    for i, (title, sub, uri) in enumerate(steps):
        t0, t1 = i / len(steps), (i + 1) / len(steps)
        vis = (f'<animate attributeName="opacity" dur="{total}s" repeatCount="indefinite" '
               f'keyTimes="0;{max(t0 - 0.0001, 0):.4f};{t0:.4f};{t1 - 0.0001:.4f};{t1:.4f};1" '
               f'values="{0 if i else 1};{0 if i else 1};1;1;0;0"/>')
        p.append(f'<g opacity="{1 if i == 0 else 0}">{vis}')
        if uri:
            p.append(f'<image x="{ix}" y="{iy}" width="{isz}" height="{isz}" href="{uri}" xlink:href="{uri}" '
                     f'style="image-rendering:pixelated"/>')
            if i == 4:
                cell = isz / 7
                for k in range(8):
                    p.append(f'<line x1="{ix + k * cell:.1f}" y1="{iy}" x2="{ix + k * cell:.1f}" y2="{iy + isz}" stroke="{ORANGE}" stroke-width="1"/>')
                    p.append(f'<line x1="{ix}" y1="{iy + k * cell:.1f}" x2="{ix + isz}" y2="{iy + k * cell:.1f}" stroke="{ORANGE}" stroke-width="1"/>')
        else:
            cell = isz / 7
            for r in range(7):
                for c in range(7):
                    inner = 0 < r < 6 and 0 < c < 6
                    fill = "#f5f5f5" if bits[r, c] else "#111"
                    p.append(f'<rect x="{ix + c * cell:.1f}" y="{iy + r * cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                             f'fill="{fill}" stroke="{ORANGE if inner else "#555"}" stroke-width="1"/>')
                    p.append(text(ix + c * cell + cell / 2, iy + r * cell + cell / 2 + 5, str(bits[r, c]), 14,
                                  "#111" if bits[r, c] else "#ddd", "middle", "700", MONO))
        p.append(text(380, 140, title, 24, TEXT, weight="800"))
        p.append(text(380, 172, sub, 14, CYAN, family=MONO))
        p.append("</g>")

    # Static explanation column and progress indicator
    lines = [
        "detect_ArUco() calls cv2.aruco.detectMarkers with DICT_5X5_50",
        "and default DetectorParameters. It returns {id: corners}, with",
        "the four corners ordered tl, tr, br, bl in the marker's own frame:",
        "the bit pattern fixes which edge is the top, so the angle is",
        "unambiguous over the full 360°.",
    ]
    for k, s in enumerate(lines):
        p.append(text(380, 214 + k * 22, s, 13, MUTED))
    for i in range(len(steps)):
        x = 380 + i * 30
        t0, t1 = i / len(steps), (i + 1) / len(steps)
        p.append(f'<rect x="{x}" y="336" width="24" height="6" rx="3" fill="{AXIS}"/>')
        p.append(f'<rect x="{x}" y="336" width="24" height="6" rx="3" fill="{CYAN}" opacity="0">'
                 f'<animate attributeName="opacity" dur="{total}s" repeatCount="indefinite" '
                 f'keyTimes="0;{t0:.4f};{t1:.4f};1" values="0;1;0;0" calcMode="discrete"/></rect>')
    save("aruco_detection.svg", p)


# ----------------------------------------------------------------------------------------
# 4. Angle geometry of Calculate_orientation over a full turn
# ----------------------------------------------------------------------------------------

def build_angle_geometry():
    W, H = 960, 400
    p = svg_open(W, H, "Angle extraction geometry")
    p.append(text(32, 40, "Calculate_orientation(): from four corners to one angle", 18, TEXT, weight="700"))
    dur = 14.0
    cx, cy, size = 230, 225, 150
    half = size / 2
    R_axis = 150

    # Axes (y points up because the code negates image y)
    p.append(f'<line x1="{cx - R_axis}" y1="{cy}" x2="{cx + R_axis}" y2="{cy}" stroke="{AXIS}" stroke-width="1.5"/>')
    p.append(f'<line x1="{cx}" y1="{cy + R_axis - 20}" x2="{cx}" y2="{cy - R_axis + 20}" stroke="{AXIS}" stroke-width="1.5"/>')
    p.append(text(cx + R_axis + 4, cy + 4, "+x", 12, MUTED))
    p.append(text(cx + 4, cy - R_axis + 16, "+y (= −image y)", 12, MUTED))

    # Rotation: math angle a from 0 to 360 => SVG rotate(90 - a)
    p.append(f'<g>'
             f'<animateTransform attributeName="transform" type="rotate" values="90 {cx} {cy};-270 {cx} {cy}" '
             f'dur="{dur}s" repeatCount="indefinite"/>')
    p.extend(marker_cells(cx, cy, size))
    p.append(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - half}" stroke="{ORANGE}" stroke-width="4"/>')
    p.append(f'<circle cx="{cx}" cy="{cy - half}" r="6" fill="{ORANGE}"/>')
    for (px, py), col, lab in [((cx - half, cy - half), MARK_TL, "tl"), ((cx + half, cy - half), MARK_TR, "tr"),
                               ((cx + half, cy + half), MARK_BR, "br"), ((cx - half, cy + half), MARK_BL, "bl")]:
        p.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{col}" stroke="#000" stroke-width="0.8"/>')
        dx = -16 if px < cx else 16
        dy = -12 if py < cy else 18
        p.append(text(px + dx, py + dy, lab, 12, TEXT, "middle", "700", MONO))
    p.append("</g>")
    p.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{MARK_CENTRE}"/>')

    # Angle arc from +x, counter-clockwise (circle flipped vertically so the stroke runs CCW)
    r_arc = 44
    circ = 2 * math.pi * r_arc
    p.append(f'<circle cx="{cx}" cy="{cy}" r="{r_arc}" fill="none" stroke="{CYAN}" stroke-width="5" '
             f'transform="translate(0 {2 * cy}) scale(1 -1)" stroke-dasharray="0 {circ:.2f}" opacity="0.9">'
             f'<animate attributeName="stroke-dasharray" values="0 {circ:.2f};{circ:.2f} 0" dur="{dur}s" repeatCount="indefinite"/>'
             f'</circle>')

    # Readout: round() is applied to arctan before adding 180 or 360, so the output is round(a), 0..360
    px0 = 470
    p.append(text(px0, 100, "angle =", 22, MUTED, weight="600"))
    p.append(f'<clipPath id="angClip"><rect x="{px0 + 88}" y="76" width="90" height="34"/></clipPath>')
    key_times = [0.0] + [(k - 0.5) / 360 for k in range(1, 361)]
    shifts = [f"0 {-k * 34}" for k in range(0, 361)]
    p.append('<g clip-path="url(#angClip)"><g>')
    p.append(f'<animateTransform attributeName="transform" type="translate" calcMode="discrete" dur="{dur}s" '
             f'repeatCount="indefinite" keyTimes="{";".join(f"{k:.5f}" for k in key_times)}" values="{";".join(shifts)}"/>')
    p.extend(tape(list(range(0, 361)), px0 + 92, 100, 34, 28, CYAN))
    p.append("</g></g>")
    p.append(text(px0 + 160, 100, "°", 24, CYAN, weight="700"))

    # Branch table, highlighted while active
    branches = [
        (0.0, 0.25, "top.x ≥ c.x, top.y ≥ c.y", "angle = arctan(dy/dx)", "0° … 90°"),
        (0.25, 0.75, "top.x < c.x", "angle = 180 + arctan(dy/dx)", "90° … 270°"),
        (0.75, 1.0, "top.x ≥ c.x, top.y < c.y", "angle = 360 + arctan(dy/dx)", "270° … 360°"),
    ]
    for i, (a0, a1, cond, rule, rng) in enumerate(branches):
        y = 150 + i * 64
        kt = f"0;{a0:.4f};{a1:.4f};1" if a0 > 0 else f"0;{a1:.4f};1"
        vals = "0.25;1;0.25;0.25" if a0 > 0 else "1;0.25;0.25"
        p.append(f'<g opacity="0.25"><animate attributeName="opacity" calcMode="discrete" dur="{dur}s" '
                 f'repeatCount="indefinite" keyTimes="{kt}" values="{vals}"/>')
        p.append(f'<rect x="{px0}" y="{y}" width="450" height="52" rx="10" fill="#16213f" stroke="{CYAN}"/>')
        p.append(text(px0 + 14, y + 21, cond, 13, TEXT, weight="600", family=MONO))
        p.append(text(px0 + 14, y + 41, rule, 13, CYAN, family=MONO))
        p.append(text(px0 + 436, y + 31, rng, 13, MUTED, "end"))
        p.append("</g>")
    p.append(text(px0, 360, "dx, dy: top-edge midpoint minus centre, with the image y axis flipped.", 12.5, MUTED))
    p.append(text(px0, 380, "Upright marker → 90°. Result rounded to whole degrees.", 12.5, MUTED))
    save("angle_geometry.svg", p)


# ----------------------------------------------------------------------------------------
# 5. Ideal damped oscillator (exact replica of ideal_harmonic_oscillator.py)
# ----------------------------------------------------------------------------------------

def ideal_response():
    I, k_rot, c_rot = 1, 15, 1
    alpha, beta = k_rot / I, c_rot / I
    A = np.array([[0, 1], [-alpha, -beta]])
    x0 = np.array([np.pi / 4, np.pi])
    tv = np.linspace(0, 10, 1000)
    states = np.array([expm(A * t) @ x0 for t in tv])
    return tv, states[:, 0], states[:, 1]


def build_oscillator():
    tv, th, om = ideal_response()
    idx = np.arange(0, len(tv), 4)
    tv, th, om = tv[idx], th[idx], om[idx]
    W, H = 960, 380
    p = svg_open(W, H, "Ideal damped rotational oscillator")
    p.append(text(32, 40, "ideal_harmonic_oscillator.py: θ(t) = C·e^(At)·x₀ with I = 1, k = 15, c = 1", 17, TEXT, weight="700"))
    dur = 10.0  # real time

    # Disk
    cx, cy, r = 150, 215, 92
    p.append(f'<line x1="{cx}" y1="70" x2="{cx}" y2="{cy}" stroke="{MUTED}" stroke-width="3" stroke-dasharray="2 3"/>')
    p.append(text(cx + 8, 84, "torsion axis", 11, MUTED))
    p.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#1b2546" stroke="{AXIS}" stroke-width="2"/>')
    p.append(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - r - 14}" stroke="{AXIS}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    rot = [f"{np.degrees(v):.2f} {cx} {cy}" for v in th]
    p.append(f'<g><animateTransform attributeName="transform" type="rotate" dur="{dur}s" repeatCount="indefinite" values="{";".join(rot)}"/>')
    p.append(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - r}" stroke="{ORANGE}" stroke-width="5" stroke-linecap="round"/>')
    p.append(f'<circle cx="{cx}" cy="{cy - r + 10}" r="6" fill="{ORANGE}"/>')
    p.append("</g>")
    p.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{TEXT}"/>')
    p.append(text(cx, 340, "θ₀ = π/4 rad, ω₀ = π rad/s", 12, MUTED, "middle"))

    # θ(t)
    pl = Plot(330, 80, 330, 230, 0, 10, -1.3, 1.3)
    p.extend(pl.frame(range(0, 11, 2), [-1, 0, 1], "time (s)", "θ (rad)"))
    env = 1.2098 * np.exp(-0.5 * tv)
    p.append(f'<path d="{pl.path(tv, env)}" fill="none" stroke="{VIOLET}" stroke-width="1" stroke-dasharray="4 4"/>')
    p.append(f'<path d="{pl.path(tv, -env)}" fill="none" stroke="{VIOLET}" stroke-width="1" stroke-dasharray="4 4"/>')
    p.append(f'<clipPath id="oscReveal"><rect x="{pl.x0}" y="{pl.y0}" width="0" height="{pl.h}">'
             f'<animate attributeName="width" values="0;{pl.w}" dur="{dur}s" repeatCount="indefinite"/></rect></clipPath>')
    p.append(f'<path d="{pl.path(tv, th)}" fill="none" stroke="{ORANGE}" stroke-width="2" clip-path="url(#oscReveal)"/>')
    p.append(f'<circle r="4.5" fill="{TEXT}"><animate attributeName="cx" values="{pl.x0};{pl.x0 + pl.w}" dur="{dur}s" repeatCount="indefinite"/>'
             f'<animate attributeName="cy" values="{fmt_values(pl.Y(th))}" dur="{dur}s" repeatCount="indefinite"/></circle>')
    p.append(text(pl.x0 + pl.w - 6, pl.y0 + 16, "envelope ±1.21·e^(−0.5t)", 11, VIOLET, "end"))

    # Phase portrait
    pp = Plot(730, 80, 200, 230, -1.3, 1.3, -4.8, 4.8)
    p.extend(pp.frame([-1, 0, 1], [-4, 0, 4], "θ (rad)", "θ̇ (rad/s)"))
    path = pp.path(th, om)
    p.append(f'<path d="{path}" fill="none" stroke="{AXIS}" stroke-width="1"/>')
    p.append(f'<circle r="5" fill="{GREEN}" filter="url(#glow)">'
             f'<animate attributeName="cx" values="{fmt_values(pp.X(th))}" dur="{dur}s" repeatCount="indefinite"/>'
             f'<animate attributeName="cy" values="{fmt_values(pp.Y(om))}" dur="{dur}s" repeatCount="indefinite"/></circle>')
    p.append(text(pp.x0 + pp.w / 2, 70, "phase portrait", 12, MUTED, "middle"))
    p.append(text(330, 365, "Matching fit on this curve (model_estimation_naive.py): β = c/2I = 0.500 s⁻¹, "
                            "ω = √(k/I − (c/2I)²) = 3.841 rad/s", 12, MUTED))
    save("oscillator.svg", p)


# ----------------------------------------------------------------------------------------
# 6. Model fitting on the real data
# ----------------------------------------------------------------------------------------

def build_fitting():
    W, H = 960, 460
    p = svg_open(W, H, "Fitting three oscillator models to the measured angle")
    p.append(text(32, 40, "Three models fitted to measured_rotation.csv (curve_fit, SciPy)", 18, TEXT, weight="700"))
    windows = [(0, 20), (150, 170)]
    models = [
        ("naive: A·e^(−βt)·cos(ωt+φ), p0 = ones", model_naive_deg, RED, SSR["naive"]),
        ("simple: + offset, peak-based p0", model_const_deg, ORANGE, SSR["constant"]),
        ("advanced: β(t), ω(t) linear in t", model_var_deg, VIOLET, SSR["variable"]),
    ]
    dur = 15.0
    plots = [Plot(80, 82, 400, 250, *windows[0], 0, 100), Plot(530, 82, 400, 250, *windows[1], 0, 100)]
    for pl, (a, b) in zip(plots, windows):
        p.extend(pl.frame(range(a, b + 1, 5), range(0, 101, 25), "time (s)", "angle (deg)" if a == 0 else ""))
        m = (time >= a) & (time <= b)
        dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5"/>' for x, y in zip(pl.X(time[m]), pl.Y(theta_deg[m])))
        p.append(f'<g fill="{CYAN}" opacity="0.85">{dots}</g>')
        for i, (_, curve, col, _) in enumerate(models):
            t_on, t_off = (1 + i * 4.3) / dur, (1 + i * 4.3 + 4.3) / dur
            d = pl.path(time[m], np.clip(curve[m], -5, 105))
            p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.4" pathLength="1" '
                     f'stroke-dasharray="1 1" stroke-dashoffset="1">'
                     f'<animate attributeName="stroke-dashoffset" dur="{dur}s" repeatCount="indefinite" '
                     f'keyTimes="0;{t_on:.3f};{t_on + 1.6 / dur:.3f};1" values="1;1;0;0"/>'
                     f'<animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" '
                     f'keyTimes="0;{t_off:.3f};{min(t_off + 0.03, 0.999):.3f};1" values="1;1;0.22;0.22"/></path>')
        p.append(text(pl.x0 + 8, pl.y0 + 18, f"t ∈ [{a}, {b}] s", 12, MUTED))

    # Legend with SSR, each row lights up with its curve
    for i, (label, _, col, ssr) in enumerate(models):
        y = 396 + i * 20
        t_on = (1 + i * 4.3) / dur
        p.append(f'<g opacity="0.3"><animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" '
                 f'keyTimes="0;{t_on:.3f};1" values="0.3;1;1" calcMode="discrete"/>')
        p.append(f'<line x1="80" y1="{y - 4}" x2="108" y2="{y - 4}" stroke="{col}" stroke-width="3"/>')
        p.append(text(118, y, label, 12.5, TEXT, family=MONO))
        p.append(text(930, y, f"SSR = {ssr:,.0f} deg²  ·  RMS = {math.sqrt(ssr / len(time)):.1f}°", 12.5, col, "end", "700", MONO))
        p.append("</g>")
    p.append(f'<circle cx="440" cy="64" r="4" fill="{CYAN}"/>')
    p.append(text(450, 68, "measurements (1 sample = 1/30 s)", 12, MUTED))
    save("fitting.svg", p)


def build_envelope():
    W, H = 960, 350
    p = svg_open(W, H, "Amplitude envelope: data vs models")
    p.append(text(32, 40, "Why a decaying model is not enough: the measured amplitude levels off", 17, TEXT, weight="700"))
    pl = Plot(80, 62, 850, 210, 0, 200, 0, 100)
    p.extend(pl.frame(range(0, 201, 25), range(0, 101, 25), "time (s)", "angle (deg)"))
    p.append(f'<path d="{pl.path(*decimate(time, theta_deg, 850))}" fill="none" stroke="{CYAN}" stroke-width="0.6" opacity="0.8"/>')
    coarse = np.linspace(0, time[-1], 300)
    env_c = np.rad2deg(P_CONST["A"] * np.exp(-P_CONST["beta"] * coarse))
    env_v = np.rad2deg(P_VAR["A"] * np.exp(-(P_VAR["beta0"] + P_VAR["beta1"] * coarse) * coarse))
    oc, ov = np.rad2deg(P_CONST["offset"]), np.rad2deg(P_VAR["offset"])
    for env, off, col in [(env_c, oc, ORANGE), (env_v, ov, VIOLET)]:
        for sgn in (1, -1):
            p.append(f'<path d="{pl.path(coarse, off + sgn * env)}" fill="none" stroke="{col}" stroke-width="2.2" '
                     f'pathLength="1" stroke-dasharray="1 1" stroke-dashoffset="1">'
                     f'<animate attributeName="stroke-dashoffset" values="1;0;0" keyTimes="0;0.5;1" dur="8s" repeatCount="indefinite"/></path>')
    # Measured half peak-to-peak per 10 s window
    centres, half = [], []
    for w0 in np.arange(0, 197, 10):
        m = (time >= w0) & (time < w0 + 10)
        centres.append(w0 + 5)
        half.append((theta_deg[m].max() - theta_deg[m].min()) / 2)
    mean = theta_deg.mean()
    for c, h in zip(centres, half):
        p.append(f'<circle cx="{pl.X(c):.1f}" cy="{pl.Y(mean + h):.1f}" r="4" fill="{TEXT}"/>')
    p.append(text(80, 318, "● measured half peak-to-peak (10 s windows)", 12, TEXT))
    p.append(text(400, 318, "— constant-parameter envelope", 12, ORANGE))
    p.append(text(640, 318, "— variable-parameter envelope", 12, VIOLET))
    p.append(text(80, 338, f"Measured amplitude: {half[0]:.1f}° at the start, {min(half):.1f}° minimum, "
                           f"{half[-1]:.1f}° at the end. Both models decay towards zero.", 12, MUTED))
    save("envelope.svg", p)


# ----------------------------------------------------------------------------------------
# 7. Kalman filter: predict/update cycle, likelihood, tracking
# ----------------------------------------------------------------------------------------

def build_kalman_cycle():
    k0, n = 300, 16
    ks = np.arange(k0, k0 + n)
    W, H = 960, 420
    p = svg_open(W, H, "Kalman predict and update cycle")
    p.append(text(32, 40, f"Kalman.py, one step at a time (real values, σq = {SIGMA_Q_BEST:.3f}, σz = {np.degrees(SIGMA_Z):.2f}°)", 17, TEXT, weight="700"))
    lo = np.degrees(min(KF["x_pred"][ks].min(), theta[ks].min())) - 14
    hi = np.degrees(max(KF["x_pred"][ks].max(), theta[ks].max())) + 14
    pl = Plot(80, 66, 560, 270, time[ks[0]] - 0.03, time[ks[-1]] + 0.03, lo, hi)
    xt = np.round(np.arange(time[ks[0]], time[ks[-1]] + 0.01, 0.1), 1)
    p.extend(pl.frame(xt, np.arange(math.ceil(lo / 10) * 10, hi, 10), "time (s)", "angle (deg)", "{:.1f}"))
    step = 1.1
    dur = step * n + 2.0

    def show(t_on):
        a = t_on / dur
        return (f'<animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" calcMode="discrete" '
                f'keyTimes="0;{a:.4f};1" values="0;1;1"/>')

    prev = None
    for i, k in enumerate(ks):
        x = pl.X(time[k])
        tp, tm, tu = i * step + 0.1, i * step + 0.45, i * step + 0.8
        xp, sp = np.degrees(KF["x_pred"][k]), np.degrees(np.sqrt(KF["P_pred"][k]))
        z = theta_deg[k]
        xe, se = np.degrees(KF["x_est"][k]), np.degrees(np.sqrt(KF["P_est"][k]))
        if prev is not None:
            p.append(f'<g opacity="0">{show(tp)}<line x1="{prev[0]:.1f}" y1="{prev[1]:.1f}" x2="{x:.1f}" y2="{pl.Y(xp):.1f}" '
                     f'stroke="{ORANGE}" stroke-width="1.4" stroke-dasharray="3 3"/></g>')
        p.append(f'<g opacity="0">{show(tp)}'
                 f'<line x1="{x - 7:.1f}" y1="{pl.Y(xp + sp):.1f}" x2="{x - 7:.1f}" y2="{pl.Y(xp - sp):.1f}" stroke="{ORANGE}" stroke-width="3" opacity="0.6"/>'
                 f'<circle cx="{x:.1f}" cy="{pl.Y(xp):.1f}" r="5" fill="none" stroke="{ORANGE}" stroke-width="2"/></g>')
        p.append(f'<g opacity="0">{show(tm)}<circle cx="{x:.1f}" cy="{pl.Y(z):.1f}" r="4.5" fill="{CYAN}"/></g>')
        p.append(f'<g opacity="0">{show(tu)}'
                 f'<line x1="{x:.1f}" y1="{pl.Y(xp):.1f}" x2="{x:.1f}" y2="{pl.Y(xe):.1f}" stroke="{PINK}" stroke-width="1.6"/>'
                 f'<line x1="{x + 7:.1f}" y1="{pl.Y(xe + se):.1f}" x2="{x + 7:.1f}" y2="{pl.Y(xe - se):.1f}" stroke="{GREEN}" stroke-width="3" opacity="0.7"/>'
                 f'<circle cx="{x:.1f}" cy="{pl.Y(xe):.1f}" r="5" fill="{GREEN}" filter="url(#glow)"/></g>')
        prev = (x, pl.Y(xe))

    # Phase indicator
    lx = 656
    rows = [
        ("Predict", "x⁻ = F_k x̂,  P⁻ = F_k P F_kᵀ + Q", ORANGE, 0.1),
        ("Measure", "z_k = θ measured by the camera", CYAN, 0.45),
        ("Innovate", "y_k = z_k − x⁻[0],  S_k = P⁻[0,0] + R", PINK, 0.8),
        ("Update", "x̂ = x⁻ + K[0]·y_k,  P = (I − K H) P⁻", GREEN, 0.8),
    ]
    for j, (name, eq, col, off) in enumerate(rows):
        y = 92 + j * 62
        keys, vals = ["0"], ["0.3"]
        for i in range(n):
            a = (i * step + off) / dur
            b = (i * step + off + 0.33) / dur
            keys += [f"{a:.4f}", f"{b:.4f}"]
            vals += ["1", "0.3"]
        keys.append("1")
        vals.append("0.3")
        p.append(f'<g opacity="0.3"><animate attributeName="opacity" calcMode="discrete" dur="{dur}s" repeatCount="indefinite" '
                 f'keyTimes="{";".join(keys)}" values="{";".join(vals)}"/>')
        p.append(f'<rect x="{lx}" y="{y - 22}" width="286" height="50" rx="10" fill="#16213f" stroke="{col}"/>')
        p.append(text(lx + 12, y - 2, name, 14, col, weight="700"))
        p.append(text(lx + 10, y + 18, eq, 10.6, TEXT, family=MONO))
        p.append("</g>")
    p.append(text(lx, 352, "F_k = [[1, dt], [0, 1 − β(t_k)·dt]]", 12, MUTED, family=MONO))
    p.append(text(80, 400, "○ prediction ±1σ   ● measurement   ● estimate ±1σ   │ correction K[0]·y_k", 12, MUTED))
    save("kalman_cycle.svg", p)


def build_likelihood():
    W, H = 960, 360
    p = svg_open(W, H, "Log-likelihood versus process noise")
    p.append(text(32, 40, "Tuning σq: the filter is run 50 times and the innovation likelihood is maximised", 17, TEXT, weight="700"))
    logq = np.log10(SIGMA_Q_GRID)
    ymin = math.floor(LIKELIHOOD.min() / 2500) * 2500
    ymax = math.ceil(LIKELIHOOD.max() / 2500) * 2500 + 1000
    pl = Plot(110, 66, 820, 210, -7, 1, ymin, ymax)
    p.extend(pl.frame(range(-7, 2), np.arange(ymin, ymax, 2500), "σq (log scale)", "log-likelihood", "1e{:d}", "{:,.0f}"))
    p.append(f'<path d="{pl.path(logq, LIKELIHOOD)}" fill="none" stroke="{AXIS}" stroke-width="2"/>')
    dur = 9.0
    xs, ys = pl.X(logq), pl.Y(LIKELIHOOD)
    for i, (x, y) in enumerate(zip(xs, ys)):
        a = (i + 1) / (len(xs) + 8)
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{CYAN}" opacity="0">'
                 f'<animate attributeName="opacity" calcMode="discrete" dur="{dur}s" repeatCount="indefinite" '
                 f'keyTimes="0;{a:.4f};1" values="0;1;1"/></circle>')
    a_best = (len(xs) + 1) / (len(xs) + 8)
    bx, by = xs[BEST], ys[BEST]
    p.append(f'<g opacity="0"><animate attributeName="opacity" calcMode="discrete" dur="{dur}s" repeatCount="indefinite" '
             f'keyTimes="0;{a_best:.4f};1" values="0;1;1"/>'
             f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="9" fill="none" stroke="{GREEN}" stroke-width="2.5" filter="url(#glow)"/>'
             f'<line x1="{bx:.1f}" y1="{by + 10:.1f}" x2="{bx:.1f}" y2="{pl.y0 + pl.h}" stroke="{GREEN}" stroke-dasharray="3 3"/>'
             + text(bx - 150, by + 30, f"σq* = {SIGMA_Q_BEST:.3f}, log L = {LIKELIHOOD[BEST]:,.1f}", 13, GREEN, "end", "700", MONO)
             + "</g>")
    p.append(text(110, 340, "log L = Σ −½ [ ln(2π S_k) + y_k² / S_k ]  over all 5926 innovations;  "
                            "σq grid = np.logspace(−7, 1, 50)", 12, MUTED, family=MONO))
    save("likelihood.svg", p)


def build_tracking():
    W, H = 960, 350
    p = svg_open(W, H, "Kalman estimates versus measurements and model")
    p.append(text(32, 40, "Where the model drifts, the filter keeps following the camera", 17, TEXT, weight="700"))
    a, b = 150, 160
    m = (time >= a) & (time <= b)
    pl = Plot(80, 62, 850, 210, a, b, 25, 75)
    p.extend(pl.frame(range(a, b + 1, 2), range(30, 71, 10), "time (s)", "angle (deg)"))
    dur = 10.0
    p.append(f'<clipPath id="trkReveal"><rect x="{pl.x0}" y="{pl.y0}" width="0" height="{pl.h}">'
             f'<animate attributeName="width" values="0;{pl.w};{pl.w}" keyTimes="0;0.8;1" dur="{dur}s" repeatCount="indefinite"/></rect></clipPath>')
    p.append(f'<g clip-path="url(#trkReveal)">')
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8"/>' for x, y in zip(pl.X(time[m]), pl.Y(theta_deg[m])))
    p.append(f'<g fill="{CYAN}">{dots}</g>')
    p.append(f'<path d="{pl.path(time[m], model_var_deg[m])}" fill="none" stroke="{VIOLET}" stroke-width="2" stroke-dasharray="6 4"/>')
    p.append(f'<path d="{pl.path(time[m], np.degrees(KF["x_est"][m]))}" fill="none" stroke="{GREEN}" stroke-width="2.2" filter="url(#glow)"/>')
    p.append("</g>")
    p.append(text(80, 318, "● measurements", 12, CYAN))
    p.append(text(200, 318, "— Kalman estimate", 12, GREEN))
    p.append(text(340, 318, "- - variable-parameter model", 12, VIOLET))
    p.append(text(80, 338, f"RMSE(estimate − measurement) = {np.degrees(RMSE_KF):.2f}° over the full record; "
                           f"σz from model residuals = {np.degrees(SIGMA_Z):.2f}°; SNR = {SNR_DB:.2f} dB", 12, MUTED))
    save("kalman_tracking.svg", p)


# ----------------------------------------------------------------------------------------
# Data bundle for the interactive lab
# ----------------------------------------------------------------------------------------

def build_lab_data():
    tv, th, _ = ideal_response()
    bundle = {
        "time": [float(v) for v in time],
        "theta_deg": [int(v) for v in theta_deg],
        "params_constant": P_CONST,
        "params_variable": P_VAR,
        "reference": {
            "ssr": SSR,
            "naive_popt": [float(v) for v in popt_naive],
            "sigma_z": SIGMA_Z,
            "sigma_q_best": SIGMA_Q_BEST,
            "log_likelihood_best": float(LIKELIHOOD[BEST]),
            "rmse_deg": float(np.degrees(RMSE_KF)),
            "snr_db": SNR_DB,
            "ideal_theta_500": float(th[500]),
        },
        "marker_bits": MARKER_BITS.tolist(),
    }
    path = os.path.join(LAB, "data.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("// Generated by docs/tools/build_figures.py from measured_rotation.csv and the parameter files.\n")
        f.write("window.MAGLEV_DATA = ")
        json.dump(bundle, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {os.path.relpath(path, ROOT)}  ({os.path.getsize(path) / 1024:.1f} kB)")


if __name__ == "__main__":
    build_hero()
    build_pipeline()
    build_detection()
    build_angle_geometry()
    build_oscillator()
    build_fitting()
    build_envelope()
    build_kalman_cycle()
    build_likelihood()
    build_tracking()
    build_lab_data()
