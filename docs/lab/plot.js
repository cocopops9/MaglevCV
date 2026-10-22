// Minimal canvas plotting used by the interactive lab (no external dependencies).

(function (root) {
  "use strict";

  const THEME = {
    panel: "#0e1530",
    grid: "#24304f",
    axis: "#3a4770",
    text: "#8d9ab8",
    font: "11px 'JetBrains Mono', SFMono-Regular, Consolas, monospace",
  };

  function niceStep(span, target) {
    const raw = span / Math.max(target, 1);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
    return nice * mag;
  }

  function ticks(lo, hi, target) {
    const step = niceStep(hi - lo, target);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
      out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return { values: out, step };
  }

  function formatTick(v, step, log) {
    if (log) return "1e" + Math.round(v);
    const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    return v.toFixed(Math.min(digits, 6));
  }

  class Plot {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.margin = Object.assign({ left: 54, right: 14, top: 14, bottom: 34 }, options.margin || {});
      this.xLabel = options.xLabel || "";
      this.yLabel = options.yLabel || "";
      this.logX = !!options.logX;
      // Pixel-column decimation is only valid when x increases along the series
      this.decimate = options.decimate !== false;
      this.height = options.height || 260;
      this.resize();
    }

    resize() {
      const dpr = window.devicePixelRatio || 1;
      const width = this.canvas.clientWidth || this.canvas.parentElement.clientWidth || 600;
      this.canvas.style.height = this.height + "px";
      this.canvas.width = Math.round(width * dpr);
      this.canvas.height = Math.round(this.height * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.w = width;
      this.h = this.height;
    }

    setRange(xmin, xmax, ymin, ymax) {
      this.xmin = xmin; this.xmax = xmax; this.ymin = ymin; this.ymax = ymax;
      return this;
    }

    X(x) {
      const v = this.logX ? Math.log10(x) : x;
      const { left, right } = this.margin;
      return left + ((v - this.xmin) / (this.xmax - this.xmin)) * (this.w - left - right);
    }

    Y(y) {
      const { top, bottom } = this.margin;
      return top + (1 - (y - this.ymin) / (this.ymax - this.ymin)) * (this.h - top - bottom);
    }

    invX(px) {
      const { left, right } = this.margin;
      const v = this.xmin + ((px - left) / (this.w - left - right)) * (this.xmax - this.xmin);
      return this.logX ? Math.pow(10, v) : v;
    }

    clear() {
      const c = this.ctx;
      const { left, right, top, bottom } = this.margin;
      c.clearRect(0, 0, this.w, this.h);
      c.fillStyle = THEME.panel;
      c.fillRect(left, top, this.w - left - right, this.h - top - bottom);
      c.font = THEME.font;
      c.lineWidth = 1;

      const xt = ticks(this.xmin, this.xmax, Math.max(3, Math.floor((this.w - left - right) / 90)));
      const yt = ticks(this.ymin, this.ymax, Math.max(3, Math.floor((this.h - top - bottom) / 50)));
      c.strokeStyle = THEME.grid;
      c.fillStyle = THEME.text;
      c.textAlign = "center";
      for (const v of xt.values) {
        const px = left + ((v - this.xmin) / (this.xmax - this.xmin)) * (this.w - left - right);
        c.beginPath(); c.moveTo(px, top); c.lineTo(px, this.h - bottom); c.stroke();
        c.fillText(formatTick(v, xt.step, this.logX), px, this.h - bottom + 15);
      }
      c.textAlign = "right";
      for (const v of yt.values) {
        const py = this.Y(v);
        c.beginPath(); c.moveTo(left, py); c.lineTo(this.w - right, py); c.stroke();
        c.fillText(formatTick(v, yt.step, false), left - 6, py + 4);
      }
      c.strokeStyle = THEME.axis;
      c.strokeRect(left + 0.5, top + 0.5, this.w - left - right - 1, this.h - top - bottom - 1);
      c.textAlign = "center";
      if (this.xLabel) c.fillText(this.xLabel, left + (this.w - left - right) / 2, this.h - 4);
      if (this.yLabel) {
        c.save();
        c.translate(12, top + (this.h - top - bottom) / 2);
        c.rotate(-Math.PI / 2);
        c.fillText(this.yLabel, 0, 0);
        c.restore();
      }
      return this;
    }

    clip(fn) {
      const c = this.ctx;
      const { left, right, top, bottom } = this.margin;
      c.save();
      c.beginPath();
      c.rect(left, top, this.w - left - right, this.h - top - bottom);
      c.clip();
      fn(c);
      c.restore();
      return this;
    }

    // Draws xs/ys between indices [i0, i1); skips sub-pixel steps for long series
    line(xs, ys, color, width = 1.5, dash = null, i0 = 0, i1 = xs.length) {
      return this.clip((c) => {
        c.strokeStyle = color;
        c.lineWidth = width;
        c.setLineDash(dash || []);
        c.beginPath();
        let lastPx = -Infinity;
        let started = false;
        let minY = Infinity, maxY = -Infinity;
        for (let i = i0; i < i1; i++) {
          const px = this.X(xs[i]);
          const py = this.Y(ys[i]);
          if (!Number.isFinite(py)) continue;
          if (!started) { c.moveTo(px, py); started = true; lastPx = px; continue; }
          if (this.decimate && px - lastPx < 0.5 && i < i1 - 1) {
            minY = Math.min(minY, py); maxY = Math.max(maxY, py);
            continue;
          }
          if (minY !== Infinity) { c.lineTo(lastPx, minY); c.lineTo(lastPx, maxY); minY = Infinity; maxY = -Infinity; }
          c.lineTo(px, py);
          lastPx = px;
        }
        c.stroke();
        c.setLineDash([]);
      });
    }

    points(xs, ys, color, r = 1.6, i0 = 0, i1 = xs.length) {
      return this.clip((c) => {
        c.fillStyle = color;
        for (let i = i0; i < i1; i++) {
          c.beginPath();
          c.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 2 * Math.PI);
          c.fill();
        }
      });
    }

    band(xs, lo, hi, color, i0 = 0, i1 = xs.length) {
      return this.clip((c) => {
        c.fillStyle = color;
        c.beginPath();
        for (let i = i0; i < i1; i++) c.lineTo(this.X(xs[i]), this.Y(hi[i]));
        for (let i = i1 - 1; i >= i0; i--) c.lineTo(this.X(xs[i]), this.Y(lo[i]));
        c.closePath();
        c.fill();
      });
    }

    vline(x, color, dash = [4, 4]) {
      return this.clip((c) => {
        c.strokeStyle = color;
        c.setLineDash(dash);
        c.beginPath();
        c.moveTo(this.X(x), this.margin.top);
        c.lineTo(this.X(x), this.h - this.margin.bottom);
        c.stroke();
        c.setLineDash([]);
      });
    }

    hline(y, color, dash = [4, 4]) {
      return this.clip((c) => {
        c.strokeStyle = color;
        c.setLineDash(dash);
        c.beginPath();
        c.moveTo(this.margin.left, this.Y(y));
        c.lineTo(this.w - this.margin.right, this.Y(y));
        c.stroke();
        c.setLineDash([]);
      });
    }

    marker(x, y, color, r = 5, filled = true) {
      return this.clip((c) => {
        c.beginPath();
        c.arc(this.X(x), this.Y(y), r, 0, 2 * Math.PI);
        if (filled) { c.fillStyle = color; c.fill(); }
        else { c.strokeStyle = color; c.lineWidth = 2; c.stroke(); }
      });
    }

    label(text, x, y, color, align = "left") {
      const c = this.ctx;
      c.fillStyle = color;
      c.font = THEME.font;
      c.textAlign = align;
      c.fillText(text, x, y);
      return this;
    }
  }

  // Index range [i0, i1) of a sorted array that falls inside [a, b]
  function windowIndices(sorted, a, b) {
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] < a) lo = m + 1; else hi = m; }
    const i0 = lo;
    hi = sorted.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] <= b) lo = m + 1; else hi = m; }
    return [i0, lo];
  }

  root.LabPlot = { Plot, windowIndices, THEME };
})(window);
