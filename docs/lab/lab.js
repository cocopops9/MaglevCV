// User interface of the interactive lab. Numerical work is delegated to models.js.

(function () {
  "use strict";

  const M = window.MaglevModels;
  const { Plot, windowIndices } = window.LabPlot;
  const D = window.MAGLEV_DATA;
  const DEG = M.DEG;

  const COLORS = {
    cyan: "#38bdf8", orange: "#f59e0b", violet: "#a78bfa", red: "#f87171",
    green: "#34d399", pink: "#f472b6", text: "#e6edf7", muted: "#8d9ab8",
  };

  const time = Float64Array.from(D.time);
  const thetaDeg = Float64Array.from(D.theta_deg);
  const thetaRad = Float64Array.from(D.theta_deg, (v) => v / DEG);

  // ---------------------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  function bindRange(id, format, onInput) {
    const input = $(id);
    const out = $(id + "-o");
    const update = () => {
      if (out) out.textContent = format(parseFloat(input.value));
      onInput(parseFloat(input.value));
    };
    input.addEventListener("input", update);
    if (out) out.textContent = format(parseFloat(input.value));
    return { input, set(v) { input.value = v; update(); }, get value() { return parseFloat(input.value); } };
  }

  function bindSegment(id, onChange) {
    const seg = $(id);
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      onChange(b.dataset.v);
    });
    return () => seg.querySelector("button.on").dataset.v;
  }

  function fmt(v, digits = 4) {
    if (!Number.isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(digits - 1);
    return v.toFixed(digits);
  }

  function rows(table, entries) {
    table.innerHTML = entries
      .map(([k, v, hl]) => `<tr class="${hl ? "hl" : ""}"><td>${k}</td><td>${v}</td></tr>`)
      .join("");
  }

  function stats(container, entries) {
    container.innerHTML = entries
      .map(([k, v, color]) => `<div class="stat"><div class="k">${k}</div><div class="v" style="color:${color || COLORS.text}">${v}</div></div>`)
      .join("");
  }

  // Deterministic random numbers, so that moving a slider does not reshuffle the noise
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    const u = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  function setupSquareCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.clientWidth || 360;
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, size };
  }

  // ---------------------------------------------------------------------------------------
  // Stepper highlighting
  // ---------------------------------------------------------------------------------------
  const navLinks = Array.from(document.querySelectorAll("nav.stepper a"));
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + entry.target.id));
      }
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  document.querySelectorAll("section.chapter").forEach((s) => observer.observe(s));

  // =======================================================================================
  // 1. Adaptive threshold explorer
  // =======================================================================================
  (function thresholdSection() {
    const N = 160;
    const bits = D.marker_bits;
    const state = { method: "adaptive", win: 13, C: 7, light: 0.55, noise: 8, rot: 20 };
    const noiseField = new Float32Array(N * N);
    const rand = mulberry32(42);
    for (let i = 0; i < noiseField.length; i++) noiseField[i] = gaussian(rand);

    const off = document.createElement("canvas");
    off.width = N; off.height = N;
    const octx = off.getContext("2d", { willReadFrequently: true });

    function renderGray() {
      octx.setTransform(1, 0, 0, 1, 0, 0);
      octx.fillStyle = "#d8d8d8";
      octx.fillRect(0, 0, N, N);
      octx.translate(N / 2, N / 2);
      octx.rotate((-state.rot * Math.PI) / 180);
      const cell = 11;
      const side = cell * 9; // 7 cells + 1 white quiet-zone cell on each side
      octx.fillStyle = "#ffffff";
      octx.fillRect(-side / 2, -side / 2, side, side);
      for (let r = 0; r < 7; r++) {
        for (let c = 0; c < 7; c++) {
          octx.fillStyle = bits[r][c] ? "#f4f4f4" : "#141414";
          octx.fillRect(-side / 2 + (c + 1) * cell - 0.25, -side / 2 + (r + 1) * cell - 0.25, cell + 0.5, cell + 0.5);
        }
      }
      const img = octx.getImageData(0, 0, N, N).data;
      const gray = new Float32Array(N * N);
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const i = y * N + x;
          // Illumination falls off linearly along the diagonal
          const illum = 1 - state.light * ((x + y) / (2 * N));
          const g = 0.299 * img[4 * i] + 0.587 * img[4 * i + 1] + 0.114 * img[4 * i + 2];
          gray[i] = Math.min(255, Math.max(0, g * illum + state.noise * noiseField[i]));
        }
      }
      return gray;
    }

    // cv2.adaptiveThreshold(ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV): replicate border, box mean
    function adaptive(gray, win, C) {
      const r = (win - 1) / 2;
      const W = N + 2 * r;
      const integral = new Float64Array((W + 1) * (W + 1));
      for (let y = 0; y < W; y++) {
        let rowSum = 0;
        const sy = Math.min(N - 1, Math.max(0, y - r));
        for (let x = 0; x < W; x++) {
          const sx = Math.min(N - 1, Math.max(0, x - r));
          rowSum += Math.round(gray[sy * N + sx]);
          integral[(y + 1) * (W + 1) + (x + 1)] = integral[y * (W + 1) + (x + 1)] + rowSum;
        }
      }
      const out = new Uint8Array(N * N);
      const area = win * win;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const x0 = x, y0 = y, x1 = x + win, y1 = y + win;
          const sum = integral[y1 * (W + 1) + x1] - integral[y0 * (W + 1) + x1] - integral[y1 * (W + 1) + x0] + integral[y0 * (W + 1) + x0];
          const T = Math.round(sum / area) - C;
          out[y * N + x] = Math.round(gray[y * N + x]) > T ? 0 : 255;
        }
      }
      return out;
    }

    function otsu(gray) {
      const hist = new Array(256).fill(0);
      gray.forEach((g) => hist[Math.round(g)]++);
      const total = gray.length;
      let sumAll = 0;
      for (let i = 0; i < 256; i++) sumAll += i * hist[i];
      let wB = 0, sumB = 0, best = 0, thr = 0;
      for (let t = 0; t < 256; t++) {
        wB += hist[t];
        if (!wB) continue;
        const wF = total - wB;
        if (!wF) break;
        sumB += t * hist[t];
        const mB = sumB / wB, mF = (sumAll - sumB) / wF;
        const between = wB * wF * (mB - mF) * (mB - mF);
        if (between > best) { best = between; thr = t; }
      }
      const out = new Uint8Array(N * N);
      for (let i = 0; i < gray.length; i++) out[i] = Math.round(gray[i]) > thr ? 0 : 255;
      return { out, thr };
    }

    function paint(canvas, values) {
      canvas.width = N; canvas.height = N;
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(N, N);
      for (let i = 0; i < N * N; i++) {
        const v = values[i];
        img.data[4 * i] = v; img.data[4 * i + 1] = v; img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }

    function render() {
      const gray = renderGray();
      paint($("thr-input"), gray);
      let binary, note;
      if (state.method === "adaptive") {
        binary = adaptive(gray, state.win, state.C);
        $("thr-title").textContent = `Adaptive mean threshold, w = ${state.win}, C = ${state.C} (inverted)`;
        note = "White = foreground (darker than local mean − C). The detector runs this for w = 3, 13 and 23 and " +
               "collects quadrilateral contours from every scale.";
      } else {
        const res = otsu(gray);
        binary = res.out;
        $("thr-title").textContent = `Global Otsu threshold = ${res.thr} (inverted)`;
        note = "A single threshold for the whole image: with strong illumination gradients the dark corner merges " +
               "with the marker border, and the quadrilateral is lost. This is why ArUco uses a local threshold.";
      }
      paint($("thr-output"), binary);
      $("thr-note").textContent = note;
    }

    bindSegment("thr-method", (v) => { state.method = v; render(); });
    bindRange("thr-win", (v) => v.toFixed(0), (v) => { state.win = v; render(); });
    bindRange("thr-c", (v) => v.toFixed(0), (v) => { state.C = v; render(); });
    bindRange("thr-light", (v) => (v * 100).toFixed(0) + " %", (v) => { state.light = v; render(); });
    bindRange("thr-noise", (v) => v.toFixed(0), (v) => { state.noise = v; render(); });
    bindRange("thr-rot", (v) => v.toFixed(0) + "°", (v) => { state.rot = v; render(); });
    render();
  })();

  // =======================================================================================
  // 2. Angle extraction playground
  // =======================================================================================
  (function angleSection() {
    const canvas = $("ang-canvas");
    const hist = new Plot($("ang-hist"), { height: 150, xLabel: "measured angle (deg)", margin: { left: 40, bottom: 32 } });
    const state = { rot: 347, tilt: 0, noise: 0 };
    const bits = D.marker_bits;
    const rand = mulberry32(7);
    let { ctx, size } = setupSquareCanvas(canvas);

    // Pinhole projection of a point (u, v) of the marker plane (v = marker up), after an
    // in-plane rotation by r and a tilt of the plane about the image x axis.
    function project(u, v) {
      const r = (state.rot * Math.PI) / 180;
      const tau = (state.tilt * Math.PI) / 180;
      const x = u * Math.cos(r) - v * Math.sin(r);
      const y = u * Math.sin(r) + v * Math.cos(r);
      const depth = 5 + y * Math.sin(tau);
      const f = size * 0.26 * 5;
      return [size / 2 + (f * x) / depth, size / 2 - (f * y * Math.cos(tau)) / depth];
    }

    function cornersFor(noiseDraw) {
      // tl, tr, br, bl in the marker's own frame, as returned by detectMarkers
      const ideal = [project(-1, 1), project(1, 1), project(1, -1), project(-1, -1)];
      if (!noiseDraw) return ideal;
      return ideal.map(([x, y]) => [x + state.noise * gaussian(rand), y + state.noise * gaussian(rand)]);
    }

    function drawMarker(corners) {
      ctx.fillStyle = "#cfd5de";
      ctx.fillRect(0, 0, size, size);
      for (let r = 0; r < 7; r++) {
        for (let c = 0; c < 7; c++) {
          const u0 = -1 + (2 * c) / 7, u1 = -1 + (2 * (c + 1)) / 7;
          const v0 = 1 - (2 * r) / 7, v1 = 1 - (2 * (r + 1)) / 7;
          const q = [project(u0, v0), project(u1, v0), project(u1, v1), project(u0, v1)];
          ctx.fillStyle = bits[r][c] ? "#f5f5f5" : "#121212";
          ctx.strokeStyle = ctx.fillStyle;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          q.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
      }
      // Overlay drawn like aruco_library.mark_ArUco (OpenCV colours converted from BGR)
      const res = M.calculateOrientation(corners);
      const top = [res.top[0], -res.top[1]];
      const centre = [res.centre[0], -res.centre[1]];
      ctx.strokeStyle = "rgb(0,24,105)";
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(top[0], top[1]); ctx.lineTo(centre[0], centre[1]); ctx.stroke();
      const colors = ["rgb(100,100,100)", "rgb(0,255,0)", "rgb(255,100,100)", "rgb(255,255,255)"];
      const names = ["tl", "tr", "br", "bl"];
      corners.forEach(([x, y], i) => {
        ctx.fillStyle = colors[i];
        ctx.strokeStyle = "#0008";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#1e293b";
        ctx.font = "600 12px 'JetBrains Mono', monospace";
        const dx = x - centre[0], dy = y - centre[1];
        const norm = Math.hypot(dx, dy) || 1;
        ctx.textAlign = "center";
        ctx.fillText(names[i], x + (dx / norm) * 18, y + (dy / norm) * 18 + 4);
      });
      ctx.fillStyle = "rgb(255,0,0)";
      ctx.beginPath(); ctx.arc(centre[0], centre[1], 5, 0, 2 * Math.PI); ctx.fill();
      ctx.textAlign = "left";
      ctx.font = "700 20px Inter, sans-serif";
      ctx.fillStyle = "rgb(200,0,0)";
      ctx.fillText("0", centre[0], centre[1]);
      ctx.fillStyle = "rgb(80,0,0)";
      ctx.fillText(String(res.angle), top[0], top[1]);

      // Reference axes of the computation (+x right, +y up)
      ctx.strokeStyle = "#47556980";
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(centre[0], centre[1]); ctx.lineTo(size - 8, centre[1]);
      ctx.stroke();
      ctx.setLineDash([]);
      return res;
    }

    function render() {
      const corners = cornersFor(state.noise > 0);
      const res = drawMarker(corners);
      const truth = (((90 + state.rot) % 360) + 360) % 360;
      let err = res.angle - truth;
      err = ((err + 540) % 360) - 180;
      rows($("ang-table"), [
        ["tl (px)", corners[0].map((v) => v.toFixed(1)).join(", ")],
        ["tr (px)", corners[1].map((v) => v.toFixed(1)).join(", ")],
        ["br (px)", corners[2].map((v) => v.toFixed(1)).join(", ")],
        ["bl (px)", corners[3].map((v) => v.toFixed(1)).join(", ")],
        ["top (y flipped)", res.top.map((v) => v.toFixed(1)).join(", ")],
        ["centre (y flipped)", res.centre.map((v) => v.toFixed(1)).join(", ")],
        ["Δx, Δy", `${res.dx.toFixed(2)}, ${res.dy.toFixed(2)}`],
        ["arctan(Δy/Δx)", `${res.raw.toFixed(3)}°`],
        ["round()", `${M.pyRound(res.raw)}°`],
        ["branch", res.branch],
        ["angle written to CSV", `${res.angle}°`, true],
        ["true 90° + r", `${truth.toFixed(1)}°`],
        ["error", `${err.toFixed(2)}°`],
      ]);

      // Monte-Carlo distribution of the measured angle under corner noise
      hist.clear();
      if (state.noise > 0) {
        const counts = new Map();
        const samples = [];
        for (let i = 0; i < 2000; i++) {
          const a = M.calculateOrientation(cornersFor(true)).angle;
          const rel = ((((a - truth) + 540) % 360) - 180);
          samples.push(rel);
          counts.set(Math.round(rel), (counts.get(Math.round(rel)) || 0) + 1);
        }
        const keys = Array.from(counts.keys());
        const lo = Math.min(...keys) - 1, hi = Math.max(...keys) + 1;
        const maxC = Math.max(...counts.values());
        hist.setRange(truth + lo, truth + hi, 0, maxC * 1.1).clear();
        hist.clip((c) => {
          c.fillStyle = COLORS.cyan;
          counts.forEach((n, k) => {
            const x0 = hist.X(truth + k - 0.4), x1 = hist.X(truth + k + 0.4);
            c.fillRect(x0, hist.Y(n), x1 - x0, hist.Y(0) - hist.Y(n));
          });
        });
        hist.vline(truth, COLORS.orange, [3, 3]);
        const m = M.mean(samples);
        const s = M.std(samples);
        $("ang-hist-cap").textContent = `2000 noisy detections: mean error ${m.toFixed(2)}°, std ${s.toFixed(2)}° (orange = true angle)`;
      } else {
        hist.setRange(truth - 5, truth + 5, 0, 1).clear();
        hist.vline(truth, COLORS.orange, [3, 3]);
        $("ang-hist-cap").textContent = "Set corner noise > 0 to see the distribution of the measured angle.";
      }
    }

    const rot = bindRange("ang-rot", (v) => v.toFixed(1) + "°", (v) => { state.rot = v; render(); });
    bindRange("ang-tilt", (v) => v.toFixed(0) + "°", (v) => { state.tilt = v; render(); });
    bindRange("ang-noise", (v) => v.toFixed(1), (v) => { state.noise = v; render(); });

    let dragging = false;
    function pointerAngle(e) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = -(e.clientY - rect.top - rect.height / 2);
      let a = (Math.atan2(y, x) * 180) / Math.PI - 90;
      a = ((a % 360) + 360) % 360;
      rot.set(a.toFixed(1));
    }
    canvas.addEventListener("pointerdown", (e) => { dragging = true; canvas.setPointerCapture(e.pointerId); pointerAngle(e); });
    canvas.addEventListener("pointermove", (e) => dragging && pointerAngle(e));
    canvas.addEventListener("pointerup", () => { dragging = false; });

    window.addEventListener("resize", () => { ({ ctx, size } = setupSquareCanvas(canvas)); hist.resize(); render(); });
    render();
  })();

  // =======================================================================================
  // 3. Oscillator simulator
  // =======================================================================================
  (function oscillatorSection() {
    const plot = new Plot($("osc-plot"), { height: 250, xLabel: "time (s)", yLabel: "θ (rad)" });
    const phase = new Plot($("osc-phase"), { height: 220, xLabel: "θ (rad)", yLabel: "θ̇ (rad/s)", decimate: false });
    const disk = $("osc-disk");
    let diskCtx = setupSquareCanvas(disk);
    const p = { I: 1, k: 15, c: 1, th0: 45, w0: 180 };
    let sim = null;
    let playing = true;
    let t0 = performance.now();
    const T_END = 10;
    const N = 1000;

    function recompute() {
      sim = M.oscillatorResponse(p.I, p.k, p.c, (p.th0 * Math.PI) / 180, (p.w0 * Math.PI) / 180, T_END, N);
      const wn = Math.sqrt(p.k / p.I);
      const zeta = p.c / (2 * Math.sqrt(p.k * p.I));
      const betaFit = p.c / (2 * p.I);
      const regime = zeta < 1 ? "under-damped" : zeta === 1 ? "critically damped" : "over-damped";
      const entries = [
        ["α = k/I", `${sim.alpha.toFixed(3)} s⁻²`],
        ["β = c/I (script)", `${sim.beta.toFixed(3)} s⁻¹`],
        ["ωₙ = √(k/I)", `${wn.toFixed(4)} rad/s`],
        ["ζ = c / 2√(kI)", zeta.toFixed(4)],
        ["regime", regime, true],
      ];
      if (zeta < 1) {
        const wd = Math.sqrt(wn * wn - betaFit * betaFit);
        const th0 = (p.th0 * Math.PI) / 180, w0 = (p.w0 * Math.PI) / 180;
        const s = -(w0 + betaFit * th0) / wd;
        entries.push(
          ["fit form: β = c/2I", `${betaFit.toFixed(4)} s⁻¹`, true],
          ["fit form: ω = ω_d", `${wd.toFixed(4)} rad/s`, true],
          ["fit form: A", `${Math.hypot(th0, s).toFixed(4)} rad`, true],
          ["fit form: φ", `${((Math.atan2(s, th0) * 180) / Math.PI).toFixed(2)}°`, true],
        );
      }
      rows($("osc-table"), entries);
    }

    function draw(tNow) {
      const idx = Math.min(N - 1, Math.round((tNow / T_END) * (N - 1)));
      const amp = Math.max(0.2, M.maxOf(sim.th.map(Math.abs)));
      plot.setRange(0, T_END, -amp * 1.15, amp * 1.15).clear();
      if (p.c > 0 && p.c / (2 * Math.sqrt(p.k * p.I)) < 1) {
        const betaFit = p.c / (2 * p.I);
        const wn = Math.sqrt(p.k / p.I);
        const wd = Math.sqrt(wn * wn - betaFit * betaFit);
        const th0 = (p.th0 * Math.PI) / 180, w0 = (p.w0 * Math.PI) / 180;
        const A0 = Math.hypot(th0, (w0 + betaFit * th0) / wd);
        const env = sim.t.map((t) => A0 * Math.exp(-betaFit * t));
        plot.line(sim.t, env, COLORS.violet, 1, [4, 4]);
        plot.line(sim.t, env.map((v) => -v), COLORS.violet, 1, [4, 4]);
      }
      plot.line(sim.t, sim.th, "#3a4770", 1);
      plot.line(sim.t, sim.th, COLORS.orange, 2, null, 0, idx + 1);
      plot.marker(sim.t[idx], sim.th[idx], COLORS.text, 4.5);

      const wmax = Math.max(0.5, M.maxOf(sim.om.map(Math.abs)));
      phase.setRange(-amp * 1.15, amp * 1.15, -wmax * 1.15, wmax * 1.15).clear();
      phase.line(sim.th, sim.om, "#3a4770", 1);
      phase.line(sim.th, sim.om, COLORS.green, 1.6, null, 0, idx + 1);
      phase.marker(sim.th[idx], sim.om[idx], COLORS.green, 5);

      const { ctx, size } = diskCtx;
      ctx.clearRect(0, 0, size, size);
      const cx = size / 2, cy = size / 2, r = size * 0.4;
      ctx.fillStyle = "#1b2546"; ctx.strokeStyle = "#3a4770"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - r - 10); ctx.stroke();
      ctx.setLineDash([]);
      const a = sim.th[idx]; // positive theta drawn counter-clockwise from the rest direction
      ctx.strokeStyle = COLORS.orange; ctx.lineWidth = 5; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx - r * Math.sin(a), cy - r * Math.cos(a)); ctx.stroke();
      ctx.fillStyle = COLORS.text;
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.font = "12px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(`t = ${sim.t[idx].toFixed(2)} s   θ = ${((a * 180) / Math.PI).toFixed(1)}°`, cx, size - 6);
    }

    let pausedAt = 0;
    function frame(now) {
      const t = playing ? (((now - t0) / 1000) % T_END) : pausedAt;
      draw(t);
      requestAnimationFrame(frame);
    }

    const controls = {
      I: bindRange("osc-I", (v) => v.toFixed(1), (v) => { p.I = v; recompute(); }),
      k: bindRange("osc-k", (v) => v.toFixed(1), (v) => { p.k = v; recompute(); }),
      c: bindRange("osc-c", (v) => v.toFixed(1), (v) => { p.c = v; recompute(); }),
      th0: bindRange("osc-th0", (v) => v.toFixed(0) + "°", (v) => { p.th0 = v; recompute(); }),
      w0: bindRange("osc-w0", (v) => v.toFixed(0) + "°/s", (v) => { p.w0 = v; recompute(); }),
    };
    $("osc-play").addEventListener("click", () => {
      if (playing) { pausedAt = ((performance.now() - t0) / 1000) % T_END; }
      else { t0 = performance.now() - pausedAt * 1000; }
      playing = !playing;
      $("osc-play").textContent = playing ? "Pause" : "Play";
    });
    $("osc-reset").addEventListener("click", () => {
      controls.I.set(1); controls.k.set(15); controls.c.set(1); controls.th0.set(45); controls.w0.set(180);
      t0 = performance.now();
    });
    window.addEventListener("resize", () => { plot.resize(); phase.resize(); diskCtx = setupSquareCanvas(disk); });
    recompute();
    requestAnimationFrame(frame);
  })();

  // =======================================================================================
  // 4. Model fitting explorer
  // =======================================================================================
  (function fittingSection() {
    const plot = new Plot($("fit-plot"), { height: 280, xLabel: "time (s)", yLabel: "angle (deg)" });
    const res = new Plot($("fit-res"), { height: 150, xLabel: "time (s)", yLabel: "residual (deg)" });

    const ideal = M.oscillatorResponse(1, 15, 1, Math.PI / 4, Math.PI, 10, 1000);
    const datasets = {
      measured: { time, theta: thetaRad, end: time[time.length - 1] },
      ideal: { time: ideal.t, theta: ideal.th, end: 10 },
    };

    const RANGES = {
      A: [0, 2.5], beta: [-0.1, 1], omega: [-0.5, 6], phi: [-Math.PI, Math.PI], offset: [-0.5, 1.6],
      beta0: [-0.06, 0.1], beta1: [-0.0006, 0.0006], omega0: [3, 4.2], omega1: [-0.01, 0.01],
    };

    const state = { data: "measured", model: "variable", p: null, peaks: null, fitter: null, running: false, start: 0, span: 30, origin: "" };

    function storedParams(model, data) {
      if (data === "ideal") return null;
      if (model === "constant") return Object.values(D.params_constant);
      if (model === "variable") return Object.values(D.params_variable);
      return D.reference.naive_popt.slice();
    }

    function initialGuess(model, data) {
      const ds = datasets[data];
      if (model === "naive") return { guess: [1, 1, 1, 1], peaks: null };
      if (model === "constant") return M.initialGuessConstant(ds.time, ds.theta);
      return M.initialGuessVariable(ds.time, ds.theta);
    }

    function buildSliders() {
      const model = M.MODELS[state.model];
      const box = $("fit-sliders");
      box.innerHTML = "";
      model.names.forEach((name, i) => {
        const [lo, hi] = RANGES[name];
        const div = document.createElement("div");
        div.className = "control";
        div.innerHTML = `<label>${name}</label><output></output><input type="range" min="${lo}" max="${hi}" step="${(hi - lo) / 2000}">`;
        const input = div.querySelector("input");
        input.addEventListener("input", () => {
          state.p[i] = parseFloat(input.value);
          state.origin = "manual";
          stopRun();
          render();
        });
        box.appendChild(div);
      });
    }

    function syncSliders() {
      const controls = $("fit-sliders").querySelectorAll(".control");
      controls.forEach((div, i) => {
        div.querySelector("input").value = state.p[i];
        div.querySelector("output").textContent = fmt(state.p[i], 5);
      });
    }

    function render() {
      const ds = datasets[state.data];
      const model = M.MODELS[state.model];
      const isMeasured = state.data === "measured";
      const end = ds.end;
      const start = Math.min(state.start, Math.max(0, end - state.span));
      const stop = Math.min(end, start + state.span);
      const [i0, i1] = windowIndices(ds.time, start, stop);
      const fitted = M.evaluate(model, ds.time, state.p);
      const yData = ds.theta.map((v) => v * DEG);
      const yFit = fitted.map((v) => v * DEG);
      const resid = yData.map((v, i) => v - yFit[i]);

      const lo = isMeasured ? 0 : -80, hi = isMeasured ? 100 : 80;
      plot.setRange(start, stop, lo, hi).clear();
      if (i1 - i0 < 900) plot.points(ds.time, yData, COLORS.cyan, 1.7, i0, i1);
      else plot.line(ds.time, yData, COLORS.cyan, 0.8, null, i0, i1);
      plot.line(ds.time, yFit, COLORS.orange, 2, null, i0, i1);
      if (state.peaks && state.peaks.length > 1) {
        const pk = state.peaks;
        [pk[0], pk[1], pk[pk.length - 2], pk[pk.length - 1]].forEach((k) => {
          if (ds.time[k] >= start && ds.time[k] <= stop) plot.marker(ds.time[k], yData[k], COLORS.text, 5, false);
        });
      }
      const rmax = Math.max(5, M.maxOf(resid.slice(i0, i1).map(Math.abs)));
      res.setRange(start, stop, -rmax * 1.1, rmax * 1.1).clear();
      res.hline(0, COLORS.muted);
      res.line(ds.time, resid, COLORS.green, 1, null, i0, i1);

      const ssr = M.ssrDegrees(model, ds.time, ds.theta, state.p);
      const entries = [
        ["SSR (deg²)", ssr.toLocaleString("en-US", { maximumFractionDigits: 1 }), COLORS.orange],
        ["RMS residual", `${Math.sqrt(ssr / ds.time.length).toFixed(3)}°`, COLORS.orange],
        ["parameters from", state.origin],
      ];
      if (state.fitter) entries.push(["LM iteration", `${state.fitter.iteration}${state.fitter.done ? " (converged)" : ""}`, COLORS.green]);
      stats($("fit-stats"), entries);
      syncSliders();
      $("fit-note").innerHTML = noteFor();
    }

    function noteFor() {
      const ref = D.reference.ssr;
      if (state.data === "ideal") {
        return state.model === "naive"
          ? "On the noise-free simulated curve, starting from p0 = ones, Levenberg–Marquardt recovers β = c/2I = 0.5 s⁻¹ and ω = 3.8406 rad/s exactly: this is the validation step of <code>model_estimation_naive.py</code>."
          : "The ideal curve has no offset and constant parameters, so the extra parameters of the richer models should converge to zero.";
      }
      if (state.model === "naive") {
        return `On the real data the naive fit fails: there is no offset term, so the optimiser trades the oscillation for a slowly varying curve. SciPy stops at ω ≈ 0 with SSR = ${Math.round(ref.naive).toLocaleString("en-US")} deg² ("Stored result"). This JS optimiser may stop at a different, equally poor, local minimum from the same start: the problem is non-convex.`;
      }
      if (state.model === "constant") {
        return `Stored result (parameters_constant.csv): SSR = ${Math.round(ref.constant).toLocaleString("en-US")} deg². The first two detected peaks have equal height (82°), so the log-decrement guess for β is exactly 0; LM still converges.`;
      }
      return `Stored result (parameters_variable.csv): SSR = ${Math.round(ref.variable).toLocaleString("en-US")} deg², 58 % lower than the constant model. Note that the phase is ω(t)·t = ω₀t + ω₁t², so the instantaneous frequency is ω₀ + 2ω₁t.`;
    }

    function load(kind) {
      stopRun();
      state.fitter = null;
      if (kind === "guess") {
        const g = initialGuess(state.model, state.data);
        state.p = g.guess.slice();
        state.peaks = g.peaks;
        state.origin = state.model === "naive" ? "p0 = ones (curve_fit default)" : "compute_initial_guess()";
      } else {
        const s = storedParams(state.model, state.data);
        if (!s) { load("guess"); return; }
        state.p = s;
        state.peaks = null;
        state.origin = state.model === "naive" ? "SciPy curve_fit result" : `parameters_${state.model}.csv`;
      }
      render();
    }

    function stopRun() {
      state.running = false;
      $("fit-run").textContent = "Run Levenberg–Marquardt";
    }

    function run() {
      if (state.running) { stopRun(); return; }
      const ds = datasets[state.data];
      state.fitter = M.createFitter(M.MODELS[state.model], ds.time, ds.theta, state.p);
      state.origin = "JS Levenberg–Marquardt";
      state.running = true;
      $("fit-run").textContent = "Stop";
      const tick = () => {
        if (!state.running) return;
        state.fitter.step();
        state.p = state.fitter.p.slice();
        render();
        if (state.fitter.done || state.fitter.iteration > 400) { stopRun(); render(); return; }
        setTimeout(() => requestAnimationFrame(tick), 90);
      };
      tick();
    }

    const getModel = bindSegment("fit-model", (v) => { state.model = v; buildSliders(); load(state.data === "ideal" ? "guess" : "stored"); });
    $("fit-data").addEventListener("change", (e) => {
      state.data = e.target.value;
      if (state.data === "ideal") { span.set(10); start.set(0); }
      else { span.set(30); }
      load(state.data === "ideal" ? "guess" : "stored");
    });
    $("fit-guess").addEventListener("click", () => load("guess"));
    $("fit-stored").addEventListener("click", () => load("stored"));
    $("fit-run").addEventListener("click", run);
    const start = bindRange("fit-start", (v) => v.toFixed(1), (v) => { state.start = v; if (state.p) render(); });
    const span = bindRange("fit-span", (v) => v.toFixed(0), (v) => { state.span = v; if (state.p) render(); });
    window.addEventListener("resize", () => { plot.resize(); res.resize(); render(); });

    state.model = getModel();
    buildSliders();
    load("stored");
  })();

  // =======================================================================================
  // 5. Kalman filter laboratory
  // =======================================================================================
  (function kalmanSection() {
    const params = D.params_variable;
    const model = M.evaluate(M.MODELS.variable, time, Object.values(params));
    const modelDeg = model.map((v) => v * DEG);
    const residuals = thetaRad.map((v, i) => v - model[i]);
    const sigmaZ = M.std(residuals); // Kalman.py: np.std(measurements - model)
    const signalPower = M.mean(thetaRad.map((v) => v * v));
    const snr = 10 * Math.log10(signalPower / (sigmaZ * sigmaZ));
    const grid = M.logspace(-7, 1, 50);

    const lik = new Plot($("kf-lik"), { height: 190, xLabel: "σq (log scale)", yLabel: "log L", logX: true, margin: { left: 60 } });
    const plot = new Plot($("kf-plot"), { height: 270, xLabel: "time (s)", yLabel: "angle (deg)" });
    const innov = new Plot($("kf-innov"), { height: 160, xLabel: "time (s)", yLabel: "innovation (deg)" });

    const state = { mode: "script", logQ: Math.log10(D.reference.sigma_q_best), start: 150, span: 8, k: 4560 };
    const sweeps = {};
    let run = null;

    function sweep(mode) {
      if (!sweeps[mode]) {
        const values = Array.from(grid, (q) => M.kalman(time, thetaRad, params, q, sigmaZ, mode, false).logL);
        let best = 0;
        values.forEach((v, i) => { if (v > values[best]) best = i; });
        sweeps[mode] = { values, best };
      }
      return sweeps[mode];
    }

    function recompute() {
      run = M.kalman(time, thetaRad, params, Math.pow(10, state.logQ), sigmaZ, state.mode, true);
      render();
    }

    function render() {
      const sw = sweep(state.mode);
      const other = sweep(state.mode === "script" ? "textbook" : "script");
      const lo = Math.min(...sw.values, ...other.values), hi = Math.max(...sw.values, ...other.values);
      lik.setRange(-7, 1, lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.12).clear();
      lik.line(grid, other.values, "#3a4770", 1.2, [4, 3]);
      lik.line(grid, sw.values, COLORS.cyan, 1.6);
      lik.points(grid, sw.values, COLORS.cyan, 2.2);
      lik.marker(grid[sw.best], sw.values[sw.best], COLORS.green, 7, false);
      lik.vline(Math.pow(10, state.logQ), COLORS.orange, [3, 3]);
      lik.marker(Math.pow(10, state.logQ), run.logL, COLORS.orange, 4.5);

      const L = run.log;
      let se = 0, mInn = 0;
      for (let i = 0; i < time.length; i++) { se += (L.xEst[i] - thetaRad[i]) ** 2; mInn += L.innov[i]; }
      mInn /= time.length;
      const sInn = M.std(L.innov);
      stats($("kf-stats"), [
        ["σz (from model residuals)", `${(sigmaZ * DEG).toFixed(3)}°`],
        ["σq", fmt(Math.pow(10, state.logQ), 4), COLORS.orange],
        ["log-likelihood", run.logL.toFixed(2), COLORS.orange],
        ["grid optimum σq*", `${fmt(grid[sw.best], 4)} (log L ${sw.values[sw.best].toFixed(1)})`, COLORS.green],
        ["RMSE(θ̂ − z)", `${(Math.sqrt(se / time.length) * DEG).toFixed(3)}°`],
        ["innovations mean / std", `${(mInn * DEG).toFixed(3)}° / ${(sInn * DEG).toFixed(3)}°`],
        ["SNR", `${snr.toFixed(2)} dB`],
      ]);

      const start = state.start, stop = Math.min(time[time.length - 1], state.start + state.span);
      const [i0, i1] = windowIndices(time, start, stop);
      const est = L.xEst.map((v) => v * DEG);
      const sd = L.PEst.map((v) => Math.sqrt(Math.max(v, 0)) * DEG);
      plot.setRange(start, stop, 15, 95).clear();
      plot.band(time, est.map((v, i) => v - sd[i]), est.map((v, i) => v + sd[i]), "rgba(52,211,153,0.22)", i0, i1);
      plot.line(time, modelDeg, COLORS.violet, 1.3, [6, 4], i0, i1);
      if (i1 - i0 < 900) plot.points(time, thetaDeg, COLORS.cyan, 1.8, i0, i1);
      else plot.line(time, thetaDeg, COLORS.cyan, 0.7, null, i0, i1);
      plot.line(time, est, COLORS.green, 1.8, null, i0, i1);
      if (time[state.k] >= start && time[state.k] <= stop) {
        plot.vline(time[state.k], COLORS.text, [2, 3]);
        plot.marker(time[state.k], L.xPred[state.k] * DEG, COLORS.orange, 5, false);
        plot.marker(time[state.k], thetaDeg[state.k], COLORS.cyan, 4.5);
        plot.marker(time[state.k], est[state.k], COLORS.green, 5);
      }

      const yInn = L.innov.map((v) => v * DEG);
      const sqS = L.S.map((v) => Math.sqrt(v) * DEG);
      const lim = Math.max(10, M.maxOf(yInn.slice(i0, i1).map(Math.abs))) * 1.1;
      innov.setRange(start, stop, -lim, lim).clear();
      innov.band(time, sqS.map((v) => -v), sqS, "rgba(244,114,182,0.15)", i0, i1);
      innov.hline(0, COLORS.muted);
      innov.line(time, yInn, COLORS.pink, 1.2, null, i0, i1);
      if (time[state.k] >= start && time[state.k] <= stop) innov.vline(time[state.k], COLORS.text, [2, 3]);

      const k = state.k;
      const dt = time[1] - time[0];
      const betaK = params.beta0 + params.beta1 * time[k];
      rows($("kf-table"), [
        ["k, t_k", `${k}, ${time[k].toFixed(4)} s`],
        ["β(t_k) = β₀ + β₁ t_k", `${fmt(betaK, 5)} s⁻¹`],
        ["F_k", `[[1, ${dt.toFixed(5)}], [0, ${(1 - betaK * dt).toFixed(6)}]]`],
        ["predicted θ⁻ / θ̇⁻", `${(L.xPred[k] * DEG).toFixed(3)}° / ${(L.vPred[k] * DEG).toFixed(2)}°/s`],
        ["P⁻[0,0]", fmt(L.PPred[k], 5)],
        ["measurement z_k", `${thetaDeg[k].toFixed(0)}°`],
        ["innovation y_k", `${(L.innov[k] * DEG).toFixed(3)}°`],
        ["S_k = P⁻[0,0] + R", fmt(L.S[k], 5)],
        ["K = [K₀, K₁]", `[${L.K0[k].toFixed(4)}, ${fmt(L.K1[k], 4)}]`],
        [state.mode === "script" ? "θ̂, θ̇̂ (both += K₀·y)" : "θ̂, θ̇̂ (+= K·y)", `${(L.xEst[k] * DEG).toFixed(3)}° / ${(L.vEst[k] * DEG).toFixed(2)}°/s`, true],
        ["P[0,0]", fmt(L.PEst[k], 5)],
      ]);
    }

    const q = bindRange("kf-q", (v) => `${v.toFixed(2)} (σq = ${fmt(Math.pow(10, v), 3)})`, (v) => { state.logQ = v; recompute(); });
    bindSegment("kf-mode", (v) => { state.mode = v; recompute(); });
    $("kf-best").addEventListener("click", () => { const sw = sweep(state.mode); q.set(Math.log10(grid[sw.best])); });
    bindRange("kf-start", (v) => v.toFixed(1), (v) => { state.start = v; render(); });
    bindRange("kf-span", (v) => v.toFixed(0), (v) => { state.span = v; render(); });
    bindRange("kf-k", (v) => `${v.toFixed(0)} (t = ${time[v].toFixed(2)} s)`, (v) => { state.k = v; render(); });
    $("kf-lik").addEventListener("click", (e) => {
      const rect = e.target.getBoundingClientRect();
      const sq = lik.invX(e.clientX - rect.left);
      q.set(Math.min(1, Math.max(-7, Math.log10(sq))));
    });
    window.addEventListener("resize", () => { lik.resize(); plot.resize(); innov.resize(); render(); });
    recompute();
  })();
})();
