// Accuracy-versus-k explorer: dataset, metric and series toggles over docs/data/curves.json.
import {
  SERIES_STYLE, FLAT_STYLE, theme, onThemeChange, loadJson, plotly, pct, el, select, checkbox,
  baseLayout, PLOT_CONFIG, mountError, onNarrowChange, phoneLayout,
} from "./wm-common.js";

const DEFAULT_SERIES = ["loma_finetuned", "loma_default", "rdd_finetuned", "rdd_default", "wildfusion"];
const DEFAULT_FLATS = ["cosine_megadescriptor", "classifier_full"];

export async function mountExplorer(root) {
  let data;
  try {
    data = await loadJson("curves.json");
  } catch (error) {
    mountError(root, error);
    return;
  }
  const datasets = { ...data.datasets, czechlynx_unseen: data.unseen };
  const state = {
    dataset: "czechlynx",
    metric: "top_5",
    series: new Set(DEFAULT_SERIES),
    flats: new Set(DEFAULT_FLATS),
  };

  root.classList.remove("wm-widget");
  root.textContent = "";
  const controls = el("div", { class: "wm-controls" });
  const plot = el("div", { class: "wm-plot", role: "img", "aria-label": "Accuracy against candidate budget" });
  const table = el("details", { class: "wm-table-view" }, [el("summary", { text: "Table view of the plotted values" })]);
  const tableBody = el("div", { class: "wm-table-wrap" });
  table.append(tableBody);
  root.append(controls, plot, table);

  const datasetOptions = Object.entries(datasets).map(([key, ds]) => [key, ds.label]);
  const metricOptions = data.metrics.map((m) => [m, data.metric_labels[m]]);
  controls.append(
    select("Dataset", datasetOptions, state.dataset, (v) => { state.dataset = v; render(); }),
    select("Metric", metricOptions, state.metric, (v) => { state.metric = v; render(); }),
  );
  const seriesBox = el("div", { class: "wm-check-group" });
  for (const key of data.series_order) {
    const sample = Object.values(datasets).map((ds) => ds.series[key]).find(Boolean);
    if (!sample) continue;
    seriesBox.append(checkbox(sample.label, state.series.has(key), (on) => {
      if (on) state.series.add(key); else state.series.delete(key);
      render();
    }, SERIES_STYLE[key].color));
  }
  const flatBox = el("div", { class: "wm-check-group" });
  for (const key of Object.keys(FLAT_STYLE)) {
    const sample = Object.values(datasets).map((ds) => ds.flats[key]).find(Boolean);
    if (!sample) continue;
    flatBox.append(checkbox(sample.label, state.flats.has(key), (on) => {
      if (on) state.flats.add(key); else state.flats.delete(key);
      render();
    }, theme().flat));
  }
  controls.append(
    el("div", { class: "wm-control wm-control--wide" }, [el("span", { text: "Matching methods" }), seriesBox]),
    el("div", { class: "wm-control wm-control--wide" }, [el("span", { text: "Budget-independent baselines" }), flatBox]),
  );

  const Plotly = await plotly();

  function render() {
    const t = theme();
    const ds = datasets[state.dataset];
    const ks = ds.ks;
    const traces = [];
    const rows = [];
    for (const key of data.series_order) {
      const entry = ds.series[key];
      if (!entry || !state.series.has(key)) continue;
      const style = SERIES_STYLE[key];
      const xs = [], ys = [];
      for (const k of ks) {
        const v = entry.points[String(k)]?.[state.metric];
        if (v != null) { xs.push(k); ys.push(v * 100); }
      }
      traces.push({
        type: "scatter", mode: "lines+markers", name: entry.label, x: xs, y: ys,
        line: { color: style.color, dash: style.dash, width: style.width, shape: "linear" },
        marker: { symbol: style.symbol, size: 9, color: style.color, line: { color: t.ring, width: 1.5 } },
        hovertemplate: "%{y:.1f} %<extra>" + entry.label + "</extra>",
      });
      rows.push([entry.label, ...ks.map((k) => pct(entry.points[String(k)]?.[state.metric]))]);
    }
    for (const key of Object.keys(FLAT_STYLE)) {
      const flat = ds.flats[key];
      if (!flat || !state.flats.has(key) || flat[state.metric] == null) continue;
      traces.push({
        type: "scatter", mode: "lines", name: flat.label, x: [ks[0], ks[ks.length - 1]],
        y: [flat[state.metric] * 100, flat[state.metric] * 100],
        line: { color: t.flat, dash: FLAT_STYLE[key].dash, width: 1.6 },
        hovertemplate: "%{y:.1f} % (independent of k)<extra>" + flat.label + "</extra>",
      });
      rows.push([flat.label, ...ks.map(() => pct(flat[state.metric]))]);
    }
    const layout = baseLayout(t, {
      xaxis: {
        ...baseLayout(t).xaxis, type: "log", tickvals: ks, ticktext: ks.map((k) => (k === 1000 ? "1k" : String(k))),
        title: { text: state.dataset === "czechlynx_unseen" ? "candidate budget k (160 = whole gallery)" : "candidate budget k", font: { color: t.ink } },
      },
      yaxis: { ...baseLayout(t).yaxis, title: { text: `${data.metric_labels[state.metric]} (%)`, font: { color: t.ink } }, rangemode: "tozero" },
      title: { text: ds.label, x: 0, font: { color: t.ink, size: 14 } },
      margin: { l: 56, r: 16, t: 72, b: 48 },
    });
    Plotly.react(plot, traces, phoneLayout(layout, { legendItems: traces.length, top: 40 }), PLOT_CONFIG);

    tableBody.textContent = "";
    const head = el("tr", {}, [el("th", { text: "Method" }), ...ks.map((k) => el("th", { text: `k = ${k}` }))]);
    const body = rows.map((r) => el("tr", {}, r.map((c, i) => el(i ? "td" : "th", { text: c, scope: i ? null : "row" }))));
    tableBody.append(el("table", { class: "wm-numeric" }, [el("thead", {}, head), el("tbody", {}, body)]));
  }

  render();
  onThemeChange(render);
  onNarrowChange(render);
}
