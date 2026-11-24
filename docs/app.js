/* Project page behaviour: equation rendering, BibTeX copy, and the attention demo. */
(function () {
  "use strict";

  // ---------------------------------------------------------------- KaTeX
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }

  // ---------------------------------------------------------------- BibTeX
  const copyButton = document.getElementById("copy-bibtex");
  if (copyButton) {
    copyButton.addEventListener("click", function () {
      const text = document.getElementById("bibtex-text").textContent;
      const done = function () {
        copyButton.textContent = "Copied";
        setTimeout(function () { copyButton.textContent = "Copy"; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      }
    });
  }

  // ---------------------------------------------------------------- Demo
  const root = document.getElementById("opro-demo");
  if (!root) return;

  const DIM = 8;            // head dimension of the toy example
  const TOKENS = 4;         // tokens per panel
  const MAX_PANELS = 4;
  const SVG_NS = "http://www.w3.org/2000/svg";

  // Deterministic pseudo-random numbers so that the figure is the same on every visit.
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rand = mulberry32(20260329);
  function gauss() {
    const u = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  // ---- small dense linear algebra on arrays of rows
  function zeros(n) { return Array.from({ length: n }, function () { return new Array(n).fill(0); }); }
  function identity(n) { const m = zeros(n); for (let i = 0; i < n; i++) m[i][i] = 1; return m; }
  function matmul(a, b) {
    const n = a.length, out = zeros(n);
    for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) {
      const v = a[i][k]; if (v === 0) continue;
      for (let j = 0; j < n; j++) out[i][j] += v * b[k][j];
    }
    return out;
  }
  function scale(a, s) { return a.map(function (row) { return row.map(function (v) { return v * s; }); }); }
  function add(a, b) { return a.map(function (row, i) { return row.map(function (v, j) { return v + b[i][j]; }); }); }
  function matvec(a, x) { return a.map(function (row) { let s = 0; for (let j = 0; j < x.length; j++) s += row[j] * x[j]; return s; }); }
  function dot(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }
  function norm(x) { return Math.sqrt(dot(x, x)); }

  // Matrix exponential by scaling and squaring with a Taylor series.
  function expm(a) {
    const n = a.length;
    let maxAbs = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) maxAbs = Math.max(maxAbs, Math.abs(a[i][j]));
    const squarings = Math.max(0, Math.ceil(Math.log2(Math.max(maxAbs * n, 1e-9))) + 2);
    const scaled = scale(a, Math.pow(2, -squarings));
    let result = identity(n), term = identity(n);
    for (let k = 1; k <= 14; k++) { term = scale(matmul(term, scaled), 1 / k); result = add(result, term); }
    for (let s = 0; s < squarings; s++) result = matmul(result, result);
    return result;
  }

  // Low-rank skew generator A = L R^T - R L^T, normalised to a fixed Frobenius norm.
  function skewGenerator(rank, frobenius) {
    const L = [], R = [];
    for (let i = 0; i < DIM; i++) { L.push([]); R.push([]); for (let r = 0; r < rank; r++) { L[i].push(gauss()); R[i].push(gauss()); } }
    const a = zeros(DIM);
    let sq = 0;
    for (let i = 0; i < DIM; i++) for (let j = 0; j < DIM; j++) {
      let v = 0;
      for (let r = 0; r < rank; r++) v += L[i][r] * R[j][r] - R[i][r] * L[j][r];
      a[i][j] = v; sq += v * v;
    }
    return scale(a, frobenius / Math.sqrt(sq));
  }

  // Fixed random content: queries, keys, generators and biases for the largest layout.
  const N_MAX = MAX_PANELS * TOKENS;
  const queries = [], keys = [];
  for (let i = 0; i < N_MAX; i++) {
    queries.push(Array.from({ length: DIM }, gauss));
    keys.push(Array.from({ length: DIM }, gauss));
  }
  const genQ = [], genK = [], biases = [];
  for (let p = 0; p < MAX_PANELS; p++) {
    genQ.push(skewGenerator(2, 2.4));
    genK.push(skewGenerator(2, 2.4));
    biases.push(Array.from({ length: DIM }, function () { return gauss() * 0.9; }));
  }

  const state = { variant: "opro", panels: 3, strength: 0.6 };

  function compute() {
    const n = state.panels * TOKENS;
    const t = state.strength;
    const opsQ = [], opsK = [];
    for (let p = 0; p < state.panels; p++) {
      opsQ.push(expm(scale(genQ[p], t)));
      opsK.push(state.variant === "asym" ? expm(scale(genK[p], t)) : opsQ[p]);
    }
    const qHat = [], kHat = [];
    let normChange = 0;
    for (let i = 0; i < n; i++) {
      const p = Math.floor(i / TOKENS);
      if (state.variant === "bias") {
        qHat.push(queries[i].map(function (v, d) { return v + t * biases[p][d]; }));
        kHat.push(keys[i].map(function (v, d) { return v + t * biases[p][d]; }));
      } else {
        qHat.push(matvec(opsQ[p], queries[i]));
        kHat.push(matvec(opsK[p], keys[i]));
      }
      normChange = Math.max(normChange,
        Math.abs(norm(qHat[i]) / norm(queries[i]) - 1), Math.abs(norm(kHat[i]) / norm(keys[i]) - 1));
    }
    const base = [], adapted = [], delta = [];
    let intra = 0, inter = 0;
    const sqrtDim = Math.sqrt(DIM);
    for (let i = 0; i < n; i++) {
      base.push([]); adapted.push([]); delta.push([]);
      for (let j = 0; j < n; j++) {
        const s0 = dot(queries[i], keys[j]) / sqrtDim;
        const s1 = dot(qHat[i], kHat[j]) / sqrtDim;
        const d = Math.abs(s1 - s0);
        base[i].push(s0); adapted[i].push(s1); delta[i].push(d);
        if (Math.floor(i / TOKENS) === Math.floor(j / TOKENS)) intra = Math.max(intra, d);
        else inter = Math.max(inter, d);
      }
    }
    return { n: n, base: base, adapted: adapted, delta: delta, intra: intra, inter: inter, normChange: normChange };
  }

  // ---- colours come from the stylesheet so that both themes work
  function cssColor(name) {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const hex = raw.replace("#", "");
    return [0, 2, 4].map(function (o) { return parseInt(hex.substr(o, 2), 16); });
  }
  function mix(a, b, t) {
    const c = a.map(function (v, i) { return Math.round(v + (b[i] - v) * t); });
    return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
  }
  let palette = null;
  function readPalette() {
    palette = {
      neg: cssColor("--heat-neg"), pos: cssColor("--heat-pos"),
      zero: cssColor("--heat-zero"), deltaMax: cssColor("--delta-max"),
    };
  }

  const LOGIT_RANGE = 2.6;   // colour scale limits, fixed so that colours are comparable
  const DELTA_RANGE = 2.2;
  function logitColor(v) {
    const t = Math.max(-1, Math.min(1, v / LOGIT_RANGE));
    return t < 0 ? mix(palette.zero, palette.neg, -t) : mix(palette.zero, palette.pos, t);
  }
  function deltaColor(v) { return mix(palette.zero, palette.deltaMax, Math.min(1, v / DELTA_RANGE)); }

  const svgs = {
    base: document.getElementById("m-base"),
    adapted: document.getElementById("m-adapted"),
    delta: document.getElementById("m-delta"),
  };

  function buildSvg(svg, n) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const pad = 0.2;
    svg.setAttribute("viewBox", (-pad) + " " + (-pad) + " " + (n + 2 * pad) + " " + (n + 2 * pad));
    const cells = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const r = document.createElementNS(SVG_NS, "rect");
      r.setAttribute("x", j + 0.04); r.setAttribute("y", i + 0.04);
      r.setAttribute("width", 0.92); r.setAttribute("height", 0.92);
      r.setAttribute("rx", 0.1);
      svg.appendChild(r); cells.push(r);
    }
    // Cross-panel frames first, same-panel frames on top.
    const panels = n / TOKENS;
    [false, true].forEach(function (diagonal) {
      for (let a = 0; a < panels; a++) for (let b = 0; b < panels; b++) {
        if ((a === b) !== diagonal) continue;
        const f = document.createElementNS(SVG_NS, "rect");
        f.setAttribute("x", b * TOKENS - 0.04); f.setAttribute("y", a * TOKENS - 0.04);
        f.setAttribute("width", TOKENS + 0.08); f.setAttribute("height", TOKENS + 0.08);
        f.setAttribute("rx", 0.22);
        f.setAttribute("class", diagonal ? "frame-intra" : "frame-inter");
        svg.appendChild(f);
      }
    });
    return cells;
  }

  let cells = null, builtFor = 0;
  const checkIntra = document.getElementById("check-intra");
  const checkNorm = document.getElementById("check-norm");
  const checkInter = document.getElementById("check-inter");
  const TOLERANCE = 5e-4;

  function render() {
    const result = compute();
    if (builtFor !== result.n) {
      cells = { base: buildSvg(svgs.base, result.n), adapted: buildSvg(svgs.adapted, result.n), delta: buildSvg(svgs.delta, result.n) };
      builtFor = result.n;
    }
    for (let i = 0; i < result.n; i++) for (let j = 0; j < result.n; j++) {
      const idx = i * result.n + j;
      cells.base[idx].setAttribute("fill", logitColor(result.base[i][j]));
      cells.adapted[idx].setAttribute("fill", logitColor(result.adapted[i][j]));
      cells.delta[idx].setAttribute("fill", deltaColor(result.delta[i][j]));
    }
    checkIntra.querySelector(".value").textContent = result.intra.toFixed(3);
    checkIntra.classList.toggle("broken", result.intra > TOLERANCE);
    checkNorm.querySelector(".value").textContent = (100 * result.normChange).toFixed(1) + "%";
    checkNorm.classList.toggle("broken", result.normChange > TOLERANCE);
    checkInter.querySelector(".value").textContent = result.inter.toFixed(3);
    document.getElementById("strength-out").textContent = state.strength.toFixed(2);
  }

  // ---- controls
  function bindGroup(attribute, apply) {
    const buttons = root.querySelectorAll("[data-" + attribute + "]");
    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        buttons.forEach(function (b) { b.setAttribute("aria-checked", String(b === button)); });
        apply(button.getAttribute("data-" + attribute));
        render();
      });
    });
  }
  bindGroup("variant", function (v) { state.variant = v; });
  bindGroup("panels", function (v) { state.panels = parseInt(v, 10); });

  const slider = document.getElementById("strength");
  slider.addEventListener("input", function () { stop(); state.strength = parseFloat(slider.value); render(); });

  const play = document.getElementById("play");
  let frame = null, started = 0;
  function stop() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null; play.textContent = "Play";
  }
  function tick(now) {
    const progress = Math.min(1, (now - started) / 3200);
    state.strength = Math.round(progress * 100) / 100;
    slider.value = String(state.strength);
    render();
    if (progress < 1) frame = requestAnimationFrame(tick); else stop();
  }
  play.addEventListener("click", function () {
    if (frame !== null) { stop(); return; }
    play.textContent = "Pause"; started = performance.now(); frame = requestAnimationFrame(tick);
  });

  const scheme = window.matchMedia("(prefers-color-scheme: dark)");
  const onScheme = function () { readPalette(); render(); };
  if (scheme.addEventListener) scheme.addEventListener("change", onScheme);

  readPalette();
  render();
})();
