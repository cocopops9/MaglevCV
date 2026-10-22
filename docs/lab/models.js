// Numerical core of the interactive lab.
// Each function is a line-by-line port of the corresponding Python code in the repository,
// so that what the lab shows is what the scripts compute. Runs in the browser and in Node
// (docs/tools/verify_lab.js checks it against the Python reference values).

(function (root) {
  "use strict";

  const DEG = 180 / Math.PI;

  // Python's built-in round(): halves go to the nearest even integer
  function pyRound(x) {
    const r = Math.round(x);
    if (Math.abs(x % 1) === 0.5) {
      return 2 * Math.round(x / 2);
    }
    return r;
  }

  // ---------------------------------------------------------------------------------------
  // aruco_library.Calculate_orientation for one marker.
  // corners: [[x, y] tl, tr, br, bl] in image pixels (y pointing down)
  // ---------------------------------------------------------------------------------------
  function calculateOrientation(corners) {
    const [tl, tr, br, bl] = corners;
    const top = [(tl[0] + tr[0]) / 2, -((tl[1] + tr[1]) / 2)];
    const centre = [(tl[0] + tr[0] + bl[0] + br[0]) / 4, -((tl[1] + tr[1] + bl[1] + br[1]) / 4)];
    const dx = top[0] - centre[0];
    const dy = top[1] - centre[1];

    // numpy float32 division by zero gives +-inf (no exception), arctan(+-inf) = +-90 deg
    let raw = Math.atan(dy / dx) * DEG;
    if (Number.isNaN(raw)) {
      raw = 0;
    }
    let angle = pyRound(raw);
    let branch = "arctan";

    if (top[0] >= centre[0] && top[1] < centre[1]) {
      angle = 360 + angle;
      branch = "360 + arctan";
    } else if (top[0] < centre[0]) {
      angle = 180 + angle;
      branch = "180 + arctan";
    }
    return { angle, raw, branch, top, centre, dx, dy };
  }

  // ---------------------------------------------------------------------------------------
  // Oscillator models (identical formulas to the model_estimation_* scripts)
  // ---------------------------------------------------------------------------------------
  const MODELS = {
    naive: {
      label: "naive",
      names: ["A", "beta", "omega", "phi"],
      f(t, p) {
        return p[0] * Math.exp(-p[1] * t) * Math.cos(p[2] * t + p[3]);
      },
      grad(t, p, g) {
        const E = Math.exp(-p[1] * t);
        const c = Math.cos(p[2] * t + p[3]);
        const s = Math.sin(p[2] * t + p[3]);
        g[0] = E * c;
        g[1] = -t * p[0] * E * c;
        g[2] = -t * p[0] * E * s;
        g[3] = -p[0] * E * s;
      },
    },
    constant: {
      label: "simple (constant parameters)",
      names: ["A", "beta", "omega", "phi", "offset"],
      f(t, p) {
        return p[0] * Math.exp(-p[1] * t) * Math.cos(p[2] * t + p[3]) + p[4];
      },
      grad(t, p, g) {
        const E = Math.exp(-p[1] * t);
        const c = Math.cos(p[2] * t + p[3]);
        const s = Math.sin(p[2] * t + p[3]);
        g[0] = E * c;
        g[1] = -t * p[0] * E * c;
        g[2] = -t * p[0] * E * s;
        g[3] = -p[0] * E * s;
        g[4] = 1;
      },
    },
    variable: {
      label: "advanced (time-varying parameters)",
      names: ["A", "beta0", "beta1", "omega0", "omega1", "phi", "offset"],
      f(t, p) {
        const beta = p[1] + p[2] * t;
        const omega = p[3] + p[4] * t;
        return p[0] * Math.exp(-beta * t) * Math.cos(omega * t + p[5]) + p[6];
      },
      grad(t, p, g) {
        const E = Math.exp(-(p[1] + p[2] * t) * t);
        const arg = (p[3] + p[4] * t) * t + p[5];
        const c = Math.cos(arg);
        const s = Math.sin(arg);
        g[0] = E * c;
        g[1] = -t * p[0] * E * c;
        g[2] = -t * t * p[0] * E * c;
        g[3] = -t * p[0] * E * s;
        g[4] = -t * t * p[0] * E * s;
        g[5] = -p[0] * E * s;
        g[6] = 1;
      },
    },
  };

  function evaluate(model, time, p) {
    const out = new Float64Array(time.length);
    for (let i = 0; i < time.length; i++) {
      out[i] = model.f(time[i], p);
    }
    return out;
  }

  // Sum of squared residuals in deg^2, as printed by the scripts
  function ssrDegrees(model, time, thetaRad, p) {
    let ssr = 0;
    for (let i = 0; i < time.length; i++) {
      const r = (thetaRad[i] - model.f(time[i], p)) * DEG;
      ssr += r * r;
    }
    return ssr;
  }

  // ---------------------------------------------------------------------------------------
  // scipy.signal.find_peaks with no arguments (scipy's _local_maxima_1d)
  // ---------------------------------------------------------------------------------------
  function findPeaks(x) {
    const peaks = [];
    let i = 1;
    const iMax = x.length - 1;
    while (i < iMax) {
      if (x[i - 1] < x[i]) {
        let iAhead = i + 1;
        while (iAhead < iMax && x[iAhead] === x[i]) {
          iAhead++;
        }
        if (x[iAhead] < x[i]) {
          const left = i;
          const right = iAhead - 1;
          peaks.push(Math.floor((left + right) / 2));
          i = iAhead;
        }
      }
      i++;
    }
    return peaks;
  }

  function mean(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return s / a.length;
  }

  function maxOf(a) {
    let m = -Infinity;
    for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
    return m;
  }

  function minOf(a) {
    let m = Infinity;
    for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i];
    return m;
  }

  // compute_initial_guess() of model_estimanion_simple.py
  function initialGuessConstant(time, theta) {
    const peaks = findPeaks(theta);
    const A1 = theta[peaks[0]];
    const A2 = theta[peaks[1]];
    const delta = Math.log(Math.abs(A1 / A2));
    const T = time[peaks[1]] - time[peaks[0]];
    const betaGuess = delta / T;
    const omegaGuess = (2 * Math.PI) / T;
    const offsetGuess = mean(theta);
    const AGuess = (maxOf(theta) - minOf(theta)) / 2;
    const phiGuess = AGuess !== 0 ? Math.acos((theta[0] - offsetGuess) / AGuess) : 0;
    return { guess: [AGuess, betaGuess, omegaGuess, phiGuess, offsetGuess], peaks };
  }

  // compute_initial_guess() of model_estimanion_advanced.py
  function initialGuessVariable(time, theta) {
    const base = initialGuessConstant(time, theta);
    const [AGuess, beta0Guess, omega0Guess, phiGuess, offsetGuess] = base.guess;
    const peaks = base.peaks;
    const last = peaks.length - 1;
    const deltaTotal = Math.log(Math.abs(theta[peaks[0]] / theta[peaks[last]]));
    const TTotal = time[peaks[last]] - time[peaks[0]];
    const betaMean = deltaTotal / TTotal;
    const beta1Guess = (2 * (betaMean - beta0Guess)) / TTotal;
    const TEnd = time[peaks[last]] - time[peaks[last - 1]];
    const omegaEnd = (2 * Math.PI) / TEnd;
    const omega1Guess = (omegaEnd - omega0Guess) / TTotal;
    return {
      guess: [AGuess, beta0Guess, beta1Guess, omega0Guess, omega1Guess, phiGuess, offsetGuess],
      peaks,
    };
  }

  // ---------------------------------------------------------------------------------------
  // Levenberg-Marquardt (the algorithm behind curve_fit's default method "lm").
  // A compact re-implementation with Marquardt's diagonal scaling; one call = one iteration.
  // ---------------------------------------------------------------------------------------
  function solveLinear(M, b) {
    const n = b.length;
    const A = M.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]];
      if (Math.abs(A[c][c]) < 1e-300) return null;
      for (let r = c + 1; r < n; r++) {
        const k = A[r][c] / A[c][c];
        for (let j = c; j <= n; j++) A[r][j] -= k * A[c][j];
      }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = A[r][n];
      for (let j = r + 1; j < n; j++) s -= A[r][j] * x[j];
      x[r] = s / A[r][r];
    }
    return x;
  }

  function createFitter(model, time, theta, p0) {
    const n = p0.length;
    const state = { p: p0.slice(), lambda: 1e-3, iteration: 0, done: false, history: [] };
    const g = new Float64Array(n);

    function sse(p) {
      let s = 0;
      for (let i = 0; i < time.length; i++) {
        const r = theta[i] - model.f(time[i], p);
        s += r * r;
      }
      return s;
    }
    state.sse = sse(state.p);

    state.step = function () {
      if (state.done) return state;
      const JtJ = Array.from({ length: n }, () => new Array(n).fill(0));
      const Jtr = new Array(n).fill(0);
      for (let i = 0; i < time.length; i++) {
        model.grad(time[i], state.p, g);
        const r = theta[i] - model.f(time[i], state.p);
        for (let a = 0; a < n; a++) {
          Jtr[a] += g[a] * r;
          for (let b = 0; b <= a; b++) JtJ[a][b] += g[a] * g[b];
        }
      }
      for (let a = 0; a < n; a++) for (let b = 0; b < a; b++) JtJ[b][a] = JtJ[a][b];

      for (let attempt = 0; attempt < 30; attempt++) {
        const M = JtJ.map((row, a) => row.map((v, b) => (a === b ? v * (1 + state.lambda) : v)));
        const delta = solveLinear(M, Jtr);
        if (!delta) {
          state.lambda *= 10;
          continue;
        }
        const trial = state.p.map((v, a) => v + delta[a]);
        const trialSse = sse(trial);
        if (Number.isFinite(trialSse) && trialSse < state.sse) {
          const improvement = (state.sse - trialSse) / state.sse;
          state.p = trial;
          state.sse = trialSse;
          state.lambda = Math.max(state.lambda / 10, 1e-12);
          state.iteration++;
          state.history.push(trialSse);
          if (improvement < 1e-12) state.done = true;
          return state;
        }
        state.lambda *= 10;
      }
      state.done = true;
      return state;
    };
    return state;
  }

  // ---------------------------------------------------------------------------------------
  // ideal_harmonic_oscillator.py: theta(t) = C exp(A t) x0 with A = [[0, 1], [-alpha, -beta]]
  // Exact 2x2 matrix exponential (Cayley-Hamilton / Putzer form).
  // ---------------------------------------------------------------------------------------
  function expm2(A, t) {
    const s = (A[0][0] + A[1][1]) / 2;
    const B = [[A[0][0] - s, A[0][1]], [A[1][0], A[1][1] - s]];
    const q2 = -(B[0][0] * B[1][1] - B[0][1] * B[1][0]); // B^2 = q2 * I
    let c, k;
    if (q2 > 1e-14) {
      const q = Math.sqrt(q2);
      c = Math.cosh(q * t);
      k = Math.sinh(q * t) / q;
    } else if (q2 < -1e-14) {
      const q = Math.sqrt(-q2);
      c = Math.cos(q * t);
      k = Math.sin(q * t) / q;
    } else {
      c = 1;
      k = t;
    }
    const e = Math.exp(s * t);
    return [
      [e * (c + k * B[0][0]), e * k * B[0][1]],
      [e * k * B[1][0], e * (c + k * B[1][1])],
    ];
  }

  function oscillatorResponse(I, kRot, cRot, theta0, omega0, tEnd, n) {
    const alpha = kRot / I;
    const beta = cRot / I;
    const A = [[0, 1], [-alpha, -beta]];
    const t = new Float64Array(n);
    const th = new Float64Array(n);
    const om = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      t[i] = (tEnd * i) / (n - 1);
      const E = expm2(A, t[i]);
      th[i] = E[0][0] * theta0 + E[0][1] * omega0;
      om[i] = E[1][0] * theta0 + E[1][1] * omega0;
    }
    return { t, th, om, alpha, beta };
  }

  // ---------------------------------------------------------------------------------------
  // Kalman.py
  // mode "script": x_est = x_pred + K[0,0] * y (the update written in Kalman.py)
  // mode "textbook": x_est = x_pred + K * y (full gain vector), for comparison
  // ---------------------------------------------------------------------------------------
  function kalman(time, z, params, sigmaQ, sigmaZ, mode, keepLog) {
    const n = time.length;
    const dt = time[1] - time[0];
    const R = sigmaZ * sigmaZ;
    const q = sigmaQ * sigmaQ;
    let x0 = z[0];
    let x1 = (z[1] - z[0]) / dt;
    let P00 = 1, P01 = 0, P10 = 0, P11 = 1;
    let logL = 0;
    const log = keepLog
      ? {
          xPred: new Float64Array(n), vPred: new Float64Array(n), PPred: new Float64Array(n),
          xEst: new Float64Array(n), vEst: new Float64Array(n), PEst: new Float64Array(n),
          innov: new Float64Array(n), S: new Float64Array(n), K0: new Float64Array(n), K1: new Float64Array(n),
        }
      : null;

    for (let k = 0; k < n; k++) {
      const betaTk = params.beta0 + params.beta1 * time[k];
      const f11 = 1 - betaTk * dt;
      // F = [[1, dt], [0, f11]]
      const xp0 = x0 + dt * x1;
      const xp1 = f11 * x1;
      // F P F^T + Q
      const a00 = P00 + dt * P10, a01 = P01 + dt * P11;
      const a10 = f11 * P10, a11 = f11 * P11;
      const pp00 = a00 + a01 * dt;
      const pp01 = a01 * f11;
      const pp10 = a10 + a11 * dt;
      const pp11 = a11 * f11 + q;

      const y = z[k] - xp0;
      const S = pp00 + R;
      const K0 = pp00 / S;
      const K1 = pp10 / S;

      if (mode === "textbook") {
        x0 = xp0 + K0 * y;
        x1 = xp1 + K1 * y;
      } else {
        x0 = xp0 + K0 * y;
        x1 = xp1 + K0 * y;
      }
      // (I - K H) P_pred with H = [1, 0]
      P00 = (1 - K0) * pp00;
      P01 = (1 - K0) * pp01;
      P10 = -K1 * pp00 + pp10;
      P11 = -K1 * pp01 + pp11;

      logL += -0.5 * (Math.log(2 * Math.PI * S) + (y * y) / S);

      if (log) {
        log.xPred[k] = xp0; log.vPred[k] = xp1; log.PPred[k] = pp00;
        log.xEst[k] = x0; log.vEst[k] = x1; log.PEst[k] = P00;
        log.innov[k] = y; log.S[k] = S; log.K0[k] = K0; log.K1[k] = K1;
      }
    }
    return { logL, log };
  }

  function logspace(a, b, n) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.pow(10, a + ((b - a) * i) / (n - 1));
    return out;
  }

  function std(a) {
    const m = mean(a);
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return Math.sqrt(s / a.length);
  }

  const api = {
    DEG, pyRound, calculateOrientation, MODELS, evaluate, ssrDegrees, findPeaks,
    initialGuessConstant, initialGuessVariable, createFitter, expm2, oscillatorResponse,
    kalman, logspace, mean, std, minOf, maxOf,
  };
  root.MaglevModels = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
