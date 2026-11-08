/* Shared helpers + SVG chart kit. Loaded before the page script in both the
   server-served UI and the static GitHub Pages build. No dependencies. */
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const f2 = v => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(2);
const pct = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (v * 100).toFixed(d) + "%";
const sign = v => v > 0 ? "up" : (v < 0 ? "down" : "flat");

function kpi(k, v, cls = "", sub = "") {
  return `<div class="kpi"><div class="k">${k}</div><div class="v ${cls}">${v}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
}

/* ---------- minimal SVG chart kit ---------- */
function svgEl(tag, attrs) {
  const el = document.createElementNS(NS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}
function grid(svg, W, H, pad, lo, hi, fmt) {
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (H - pad.t - pad.b) * i / 4;
    svg.appendChild(svgEl("line", { x1: pad.l, y1: y, x2: W - pad.r, y2: y, stroke: "#eef0f4" }));
    const label = svgEl("text", { x: pad.l - 10, y: y + 4, "font-size": 11, fill: "#98a2b3", "text-anchor": "end" });
    label.textContent = fmt(hi - (hi - lo) * i / 4);
    svg.appendChild(label);
  }
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

function drawCandles(svg, bars) {
  clear(svg);
  const W = 960, H = 340, pad = { l: 62, r: 18, t: 18, b: 30 };
  let lo = Math.min(...bars.map(b => b.low)), hi = Math.max(...bars.map(b => b.high));
  const span = (hi - lo) || 1; lo -= span * .05; hi += span * .05;
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const step = plotW / bars.length, bw = Math.max(step * .62, 1);
  const Y = v => pad.t + plotH * (1 - (v - lo) / (hi - lo));

  grid(svg, W, H, pad, lo, hi, v => v.toFixed(1));
  bars.forEach((b, i) => {
    const x = pad.l + step * i + step / 2;
    const color = b.close >= b.open ? "#c0392b" : "#0e7c6b";
    svg.appendChild(svgEl("line", { x1: x, y1: Y(b.high), x2: x, y2: Y(b.low), stroke: color, "stroke-width": 1 }));
    const top = Y(Math.max(b.open, b.close)), bot = Y(Math.min(b.open, b.close));
    svg.appendChild(svgEl("rect", { x: x - bw / 2, y: top, width: bw, height: Math.max(bot - top, 1), fill: color, opacity: .88, rx: .5 }));
  });
  const t1 = svgEl("text", { x: pad.l, y: H - 8, "font-size": 11, fill: "#98a2b3" });
  t1.textContent = bars[0].date; svg.appendChild(t1);
  const t2 = svgEl("text", { x: W - pad.r, y: H - 8, "font-size": 11, fill: "#98a2b3", "text-anchor": "end" });
  t2.textContent = bars[bars.length - 1].date; svg.appendChild(t2);
}

function drawLines(svg, series, opts = {}) {
  clear(svg);
  const W = 960, H = opts.height || 340, pad = { l: 62, r: 18, t: 18, b: 30 };
  const all = series.flatMap(s => s.values).filter(v => typeof v === "number");
  let lo = Math.min(...all), hi = Math.max(...all);
  if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const span = (hi - lo) || 1; lo -= span * .06; hi += span * .06;
  const n = Math.max(...series.map(s => s.values.length));
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const X = i => pad.l + plotW * (n <= 1 ? 0 : i / (n - 1));
  const Y = v => pad.t + plotH * (1 - (v - lo) / (hi - lo));

  grid(svg, W, H, pad, lo, hi, v => Number(v).toFixed(2));
  if (opts.zero) svg.appendChild(svgEl("line", {
    x1: pad.l, y1: Y(0), x2: W - pad.r, y2: Y(0), stroke: "#cfd6e4", "stroke-dasharray": "4 4"
  }));
  series.forEach(s => {
    const d = s.values.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    svg.appendChild(svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 1.9, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  });
}

function drawBars(svg, values, labels, opts = {}) {
  clear(svg);
  const W = 960, H = opts.height || 200, pad = { l: 62, r: 18, t: 18, b: 34 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const max = Math.max(...values.map(Math.abs)) || 1;
  const zero = pad.t + plotH * (opts.zero ? .5 : 0);
  const step = plotW / values.length;
  grid(svg, W, H, pad, 0, max, v => (opts.pct ? (v * 100).toFixed(1) + "%" : v.toFixed(3)));
  if (opts.zero) svg.appendChild(svgEl("line", { x1: pad.l, y1: H - pad.b, x2: W - pad.r, y2: H - pad.b, stroke: "#cfd6e4" }));
  values.forEach((v, i) => {
    const h = Math.max((Math.abs(v) / max) * (plotH - 8), 1);
    const x = pad.l + step * i + step * .18;
    const y = v >= 0 ? (H - pad.b) - h : (H - pad.b);
    svg.appendChild(svgEl("rect", {
      x, y, width: step * .64, height: h, rx: 4,
      fill: v >= 0 ? "#c0392b" : "#0e7c6b", opacity: .85
    }));
    const t = svgEl("text", { x: x + step * .32, y: H - 14, "font-size": 11, fill: "#98a2b3", "text-anchor": "middle" });
    t.textContent = labels[i]; svg.appendChild(t);
  });
}
