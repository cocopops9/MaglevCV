// Checks the JavaScript ports in docs/lab/models.js against the reference values that
// docs/tools/build_figures.py computed with the Python code. Run from the repository root:
//     node docs/tools/verify_lab.js

"use strict";

const path = require("path");
globalThis.window = globalThis;
require(path.join(__dirname, "..", "lab", "data.js"));
const M = require(path.join(__dirname, "..", "lab", "models.js"));

const D = globalThis.MAGLEV_DATA;
const time = Float64Array.from(D.time);
const theta = Float64Array.from(D.theta_deg, (v) => v / M.DEG);
let failures = 0;

function check(name, got, want, relTol) {
  const rel = Math.abs(got - want) / Math.max(Math.abs(want), 1e-12);
  const ok = rel <= relTol;
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: js = ${got.toPrecision(10)}, python = ${Number(want).toPrecision(10)}`);
}

// 1. Orientation: a marker rotated counter-clockwise by r degrees must read round(90 + r) mod 360
for (const r of [0, 13, 45, 89.4, 90, 133, 180, 246, 270, 300, 359]) {
  const a = ((90 + r) * Math.PI) / 180;
  const c = [320, 240];
  const h = 50;
  const ux = [Math.cos(a), -Math.sin(a)]; // marker "up" direction in image coordinates (y down)
  const rx = [Math.sin(a), Math.cos(a)];  // marker "right" direction
  const corner = (u, v) => [c[0] + h * (v * rx[0] + u * ux[0]), c[1] + h * (v * rx[1] + u * ux[1])];
  const res = M.calculateOrientation([corner(1, -1), corner(1, 1), corner(-1, 1), corner(-1, -1)]);
  const want = (Math.round(90 + r) % 360);
  const ok = res.angle % 360 === want;
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} orientation rotate ${r} deg -> ${res.angle} (${res.branch})`);
}

// 2. SSR of the stored parameter files
const pc = Object.values(D.params_constant);
const pv = Object.values(D.params_variable);
check("SSR constant", M.ssrDegrees(M.MODELS.constant, time, theta, pc), D.reference.ssr.constant, 1e-9);
check("SSR variable", M.ssrDegrees(M.MODELS.variable, time, theta, pv), D.reference.ssr.variable, 1e-9);
check("SSR naive", M.ssrDegrees(M.MODELS.naive, time, theta, D.reference.naive_popt), D.reference.ssr.naive, 1e-9);

// 3. Initial guesses (values printed by the Python port of compute_initial_guess)
const g1 = M.initialGuessConstant(time, theta);
check("peaks found", g1.peaks.length, 246, 0);
check("guess A", g1.guess[0], 0.5235987755982989, 1e-12);
check("guess omega", g1.guess[2], 3.6249146002959156, 1e-12);
check("guess phi", g1.guess[3], 0.4382767811584651, 1e-12);
const g2 = M.initialGuessVariable(time, theta);
check("guess beta1", g2.guess[2], 1.4565553575835887e-05, 1e-9);
check("guess omega1", g2.guess[4], 0.0049636172005478076, 1e-9);

// 4. Levenberg-Marquardt from the scripts' initial guesses reaches the stored optimum
for (const [name, model, guess, stored] of [
  ["constant", M.MODELS.constant, g1.guess, pc],
  ["variable", M.MODELS.variable, g2.guess, pv],
]) {
  const fit = M.createFitter(model, time, theta, guess);
  while (!fit.done && fit.iteration < 500) fit.step();
  check(`LM ${name} SSR after ${fit.iteration} iterations`, M.ssrDegrees(model, time, theta, fit.p),
        M.ssrDegrees(model, time, theta, stored), 1e-6);
  fit.p.forEach((v, i) => check(`LM ${name} ${model.names[i]}`, v, stored[i], 2e-3));
}

// 5. Kalman filter: sigma_z, likelihood sweep, optimum, RMSE
const residuals = M.evaluate(M.MODELS.variable, time, pv).map((m, i) => theta[i] - m);
const sigmaZ = M.std(residuals);
check("sigma_z", sigmaZ, D.reference.sigma_z, 1e-12);
const grid = M.logspace(-7, 1, 50);
let best = -Infinity, bestQ = 0;
for (const q of grid) {
  const { logL } = M.kalman(time, theta, D.params_variable, q, sigmaZ, "script", false);
  if (logL > best) { best = logL; bestQ = q; }
}
check("best sigma_q", bestQ, D.reference.sigma_q_best, 1e-9);
check("max log-likelihood", best, D.reference.log_likelihood_best, 1e-9);
const run = M.kalman(time, theta, D.params_variable, bestQ, sigmaZ, "script", true);
let se = 0;
for (let i = 0; i < time.length; i++) se += (run.log.xEst[i] - theta[i]) ** 2;
check("RMSE deg", Math.sqrt(se / time.length) * M.DEG, D.reference.rmse_deg, 1e-9);

// 6. Matrix exponential response of ideal_harmonic_oscillator.py at t = 10 * 500 / 999
const osc = M.oscillatorResponse(1, 15, 1, Math.PI / 4, Math.PI, 10, 1000);
check("oscillator theta[500]", osc.th[500], D.reference.ideal_theta_500, 1e-9);

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
