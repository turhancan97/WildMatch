// Shared helpers for the project-page components: brand palette, theme detection,
// data loading relative to the site root, Plotly bootstrapping and small DOM utilities.

export const PALETTE = {
  blue: "#3a7eab", red: "#cf4832", gold: "#b8860b",
  greyText: "#6d6e71", greyHead: "#58595b", greyRule: "#d1d3d4",
};

// Colour follows the matcher family; the fine-tuned series is solid with filled
// markers and the default dashed with hollow markers, so hue is never the only cue.
export const SERIES_STYLE = {
  loma_finetuned: { color: PALETTE.blue, dash: "solid", symbol: "circle", width: 2.6 },
  loma_default: { color: PALETTE.blue, dash: "dash", symbol: "circle-open", width: 2 },
  rdd_finetuned: { color: PALETTE.red, dash: "solid", symbol: "square", width: 2.6 },
  rdd_default: { color: PALETTE.red, dash: "dash", symbol: "square-open", width: 2 },
  wildfusion: { color: PALETTE.gold, dash: "dot", symbol: "triangle-up", width: 2 },
};

export const FLAT_STYLE = {
  cosine_megadescriptor: { dash: "dot" },
  cosine_dinov3: { dash: "dashdot" },
  classifier_full: { dash: "solid" },
  classifier_partial: { dash: "dash" },
  classifier_frozen: { dash: "longdashdot" },
};

export function isDark() {
  return document.body.getAttribute("data-md-color-scheme") === "slate";
}

export function theme() {
  const dark = isDark();
  return {
    dark,
    ink: dark ? "#e6e7e8" : PALETTE.greyHead,
    muted: dark ? "#b3b5b8" : PALETTE.greyText,
    grid: dark ? "#3b3f44" : "#e7e8e9",
    ring: dark ? "#1e2227" : "#ffffff",
    flat: dark ? "#b3b5b8" : PALETTE.greyText,
  };
}

export function onThemeChange(callback) {
  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.attributeName === "data-md-color-scheme") { callback(); return; }
    }
  });
  observer.observe(document.body, { attributes: true });
  return observer;
}

// docs/assets/js/<module>.js -> docs/data/<name>; works under any site_url prefix.
export function dataUrl(name) {
  return new URL(`../../data/${name}`, import.meta.url).href;
}

export function assetUrl(path) {
  return new URL(`../${path}`, import.meta.url).href;
}

const cache = new Map();
export async function loadJson(name) {
  if (!cache.has(name)) {
    cache.set(name, fetch(dataUrl(name)).then((r) => {
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(name);
}

export function plotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  return new Promise((resolve, reject) => {
    let waited = 0;
    const timer = setInterval(() => {
      if (window.Plotly) { clearInterval(timer); resolve(window.Plotly); }
      else if ((waited += 100) > 15000) { clearInterval(timer); reject(new Error("Plotly did not load")); }
    }, 100);
  });
}

export function pct(value, digits = 1) {
  return value == null || Number.isNaN(value) ? "–" : (value * 100).toFixed(digits);
}

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child != null) node.append(child);
  }
  return node;
}

export function select(label, options, value, onChange) {
  const sel = el("select", { class: "wm-select", "aria-label": label });
  for (const [val, text] of options) sel.append(el("option", { value: val, text }));
  sel.value = value;
  sel.addEventListener("change", () => onChange(sel.value));
  return el("label", { class: "wm-control" }, [el("span", { text: label }), sel]);
}

export function checkbox(label, checked, onChange, swatch) {
  const box = el("input", { type: "checkbox" });
  box.checked = checked;
  box.addEventListener("change", () => onChange(box.checked));
  const children = [box];
  if (swatch) children.push(el("span", { class: "wm-swatch", style: `--wm-swatch:${swatch}` }));
  children.push(el("span", { text: label }));
  return el("label", { class: "wm-check" }, children);
}

export function baseLayout(t, overrides = {}) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Roboto, Helvetica, Arial, sans-serif", size: 12, color: t.ink },
    margin: { l: 56, r: 16, t: 24, b: 48 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: t.dark ? "#2b3036" : "#ffffff", bordercolor: t.grid, font: { color: t.ink } },
    legend: { orientation: "h", y: 1.12, x: 0, font: { color: t.ink } },
    xaxis: { gridcolor: t.grid, linecolor: t.grid, zeroline: false, tickfont: { color: t.muted }, title: { font: { color: t.ink } } },
    yaxis: { gridcolor: t.grid, linecolor: t.grid, zeroline: false, tickfont: { color: t.muted }, title: { font: { color: t.ink } } },
    ...overrides,
  };
}

export const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] };

// Phone layout. At this width a horizontal legend in the top margin wraps into several rows
// and covers the plot, and side-by-side panels get too narrow to read (checked at 390 px).
const NARROW_QUERY = "(max-width: 600px)";

export function isNarrow() {
  return window.matchMedia(NARROW_QUERY).matches;
}

// Re-render when the viewport crosses the phone breakpoint (e.g. a rotated phone).
export function onNarrowChange(callback) {
  window.matchMedia(NARROW_QUERY).addEventListener("change", callback);
}

// On phones: legend as a vertical list under the plot, the figure made taller by exactly the
// legend's height so the plot area keeps its size. Wide layouts are returned unchanged.
export function phoneLayout(layout, { legendItems = 0, height = 380, top } = {}) {
  if (!isNarrow()) return layout;
  const legendHeight = legendItems ? legendItems * 19 + 12 : 0;
  const margin = { ...layout.margin };
  margin.b = (margin.b ?? 48) + legendHeight;
  if (top != null) margin.t = top;
  return {
    ...layout,
    height: height + legendHeight,
    margin,
    legend: { ...layout.legend, orientation: "v", x: 0, xanchor: "left", y: 0, yanchor: "bottom", yref: "container" },
  };
}

export function mountError(root, error) {
  root.classList.add("wm-widget--error");
  root.textContent = `This view could not load its data (${error.message}).`;
}
