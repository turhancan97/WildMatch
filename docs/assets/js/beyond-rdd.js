// Beyond the paper: RDD-LightGlue retrained with the shared recipe against the paper's fine-tuned and
// default RDD-LightGlue, over docs/data/beyond.json. All three are RDD (red); line style and marker
// tell them apart: retrained solid with filled diamonds, the paper's fine-tuned dotted with filled
// squares, the default dashed with hollow squares.
import {
  PALETTE, theme, onThemeChange, loadJson, plotly, pct, el, select,
  baseLayout, PLOT_CONFIG, mountError, onNarrowChange, phoneLayout,
} from "./wm-common.js";

const STYLE = {
  rdd_retrained: { dash: "solid", symbol: "diamond", width: 2.8 },
  rdd_paper: { dash: "dot", symbol: "square", width: 2.2 },
  rdd_default: { dash: "dash", symbol: "square-open", width: 2 },
};

export async function mountBeyondRdd(root) {
  let data;
  try {
    data = await loadJson("beyond.json");
  } catch (error) {
    mountError(root, error);
    return;
  }
  const rdd = data.rdd;
  const state = { dataset: Object.keys(rdd.datasets)[0], metric: "top_1" };

  root.classList.remove("wm-widget");
  root.textContent = "";
  const controls = el("div", { class: "wm-controls" });
  const plot = el("div", { class: "wm-plot", role: "img", "aria-label": "RDD-LightGlue accuracy against candidate budget" });
  const table = el("details", { class: "wm-table-view" }, [el("summary", { text: "Table view of the plotted values" })]);
  const tableBody = el("div", { class: "wm-table-wrap" });
  table.append(tableBody);
  root.append(controls, plot, table);
  controls.append(
    select("Dataset", Object.entries(rdd.datasets).map(([key, ds]) => [key, ds.label]), state.dataset, (v) => {
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
    const ds = rdd.datasets[state.dataset];
    const ks = ds.ks;
    const traces = [];
    const rows = [];
    for (const key of rdd.series_order) {
      const points = ds.series[key];
      const style = STYLE[key];
      const label = rdd.series_labels[key];
      const xs = [], ys = [];
      for (const k of ks) {
        const v = points[String(k)]?.[state.metric];
        if (v != null) { xs.push(k); ys.push(v * 100); }
      }
      traces.push({
        type: "scatter", mode: "lines+markers", name: label, x: xs, y: ys,
        line: { color: PALETTE.red, dash: style.dash, width: style.width, shape: "linear" },
        marker: { symbol: style.symbol, size: 9, color: PALETTE.red, line: { color: t.ring, width: 1.5 } },
        hovertemplate: "%{y:.1f} %<extra>" + label + "</extra>",
      });
      rows.push([label, ...ks.map((k) => pct(points[String(k)]?.[state.metric]))]);
    }
    const unseen = state.dataset === "czechlynx_unseen";
    const layout = baseLayout(t, {
      xaxis: {
        ...baseLayout(t).xaxis, type: "log", tickvals: ks, ticktext: ks.map((k) => (k === 1000 ? "1k" : String(k))),
        title: { text: unseen ? "candidate budget k (160 = whole gallery)" : "candidate budget k", font: { color: t.ink } },
      },
      yaxis: { ...baseLayout(t).yaxis, title: { text: `${data.metric_labels[state.metric]} (%)`, font: { color: t.ink } }, rangemode: "tozero" },
      title: { text: ds.label, x: 0, font: { color: t.ink, size: 14 } },
      margin: { l: 56, r: 16, t: 72, b: 48 },
    });
    Plotly.react(plot, traces, phoneLayout(layout, { legendItems: traces.length, top: 40 }), PLOT_CONFIG);

    tableBody.textContent = "";
    const head = el("tr", {}, [el("th", { text: "Checkpoint" }), ...ks.map((k) => el("th", { text: `k = ${k}` }))]);
    const body = rows.map((r) => el("tr", {}, r.map((c, i) => el(i ? "td" : "th", { text: c, scope: i ? null : "row" }))));
    tableBody.append(el("table", { class: "wm-numeric" }, [el("thead", {}, head), el("tbody", {}, body)]));
  }

  render();
  onThemeChange(render);
  onNarrowChange(render);
}
