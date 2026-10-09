// Matcher ablation: four matchers with their default weights against the candidate budget, over
// docs/data/matcher_ablation.json. LoMa and RDD-LightGlue keep their family colours; the two extra
// matchers are grey and told apart by line style and marker, never by hue alone.
import {
  SERIES_STYLE, theme, onThemeChange, loadJson, plotly, pct, el, select,
  baseLayout, PLOT_CONFIG, mountError, onNarrowChange, phoneLayout,
} from "./wm-common.js";

function seriesStyle(key, t) {
  if (key === "loma_default" || key === "rdd_default") return SERIES_STYLE[key];
  if (key === "superpoint_default") return { color: t.ink, dash: "dot", symbol: "diamond-open", width: 2 };
  return { color: t.muted, dash: "dashdot", symbol: "triangle-up-open", width: 2 };
}

export async function mountMatcherAblation(root) {
  let data;
  try {
    data = await loadJson("matcher_ablation.json");
  } catch (error) {
    mountError(root, error);
    return;
  }
  const datasets = { mean: data.mean, ...data.datasets };
  const state = { dataset: "mean", metric: "top_1" };

  root.classList.remove("wm-widget");
  root.textContent = "";
  const controls = el("div", { class: "wm-controls" });
  const plot = el("div", { class: "wm-plot", role: "img", "aria-label": "Accuracy of four matchers against candidate budget" });
  const table = el("details", { class: "wm-table-view" }, [el("summary", { text: "Table view of the plotted values" })]);
  const tableBody = el("div", { class: "wm-table-wrap" });
  table.append(tableBody);
  root.append(controls, plot, table);

  controls.append(
    select("Dataset", Object.entries(datasets).map(([key, ds]) => [key, ds.label]), state.dataset, (v) => {
      state.dataset = v;
      render();
    }),
    select("Metric", data.metrics.map((m) => [m, data.metric_labels[m]]), state.metric, (v) => {
      state.metric = v;
      render();
    }),
  );

  const Plotly = await plotly();

  function render() {
    const t = theme();
    const ds = datasets[state.dataset];
    const ks = data.ks;
    const traces = [];
    const rows = [];
    for (const key of data.series_order) {
      const entry = ds.series[key];
      const style = seriesStyle(key, t);
      const xs = [], ys = [];
      for (const k of ks) {
        const v = entry.points[String(k)]?.[state.metric];
        if (v != null) { xs.push(k); ys.push(v * 100); }
      }
      traces.push({
        type: "scatter", mode: "lines+markers", name: entry.label, x: xs, y: ys,
        line: { color: style.color, dash: style.dash, width: style.width, shape: "linear" },
        marker: { symbol: style.symbol, size: 9, color: style.color, line: { color: style.color, width: 1.5 } },
        hovertemplate: "%{y:.1f} %<extra>" + entry.label + "</extra>",
      });
      rows.push([entry.label, ...ks.map((k) => pct(entry.points[String(k)]?.[state.metric]))]);
    }
    const layout = baseLayout(t, {
      xaxis: {
        ...baseLayout(t).xaxis, type: "log", tickvals: ks, ticktext: ks.map((k) => (k === 1000 ? "1k" : String(k))),
        title: { text: "candidate budget k", font: { color: t.ink } },
      },
      yaxis: { ...baseLayout(t).yaxis, title: { text: `${data.metric_labels[state.metric]} (%)`, font: { color: t.ink } }, rangemode: "tozero" },
      title: { text: ds.label, x: 0, font: { color: t.ink, size: 14 } },
      margin: { l: 56, r: 16, t: 72, b: 48 },
    });
    Plotly.react(plot, traces, phoneLayout(layout, { legendItems: traces.length, top: 40 }), PLOT_CONFIG);

    tableBody.textContent = "";
    const head = el("tr", {}, [el("th", { text: "Matcher" }), ...ks.map((k) => el("th", { text: `k = ${k}` }))]);
    const body = rows.map((r) => el("tr", {}, r.map((c, i) => el(i ? "td" : "th", { text: c, scope: i ? null : "row" }))));
    tableBody.append(el("table", { class: "wm-numeric" }, [el("thead", {}, head), el("tbody", {}, body)]));
  }

  render();
  onThemeChange(render);
  onNarrowChange(render);
}
