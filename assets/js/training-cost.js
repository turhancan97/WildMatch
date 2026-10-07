// Training cost against accuracy (CzechLynx closed, k = 250) over docs/data/training_cost.json.
import {
  PALETTE, theme, onThemeChange, loadJson, plotly, pct, el, select, baseLayout, PLOT_CONFIG, mountError, onNarrowChange, phoneLayout,
} from "./wm-common.js";

const CLASSIFIER_DASH = { classifier_full: "solid", classifier_partial: "dash", classifier_frozen: "dashdot" };
const METRICS = [["top_5", "Top-5"], ["balanced_top_1", "Balanced top-1"], ["top_1", "Top-1"]];

export async function mountTrainingCost(root) {
  let data;
  try {
    data = await loadJson("training_cost.json");
  } catch (error) {
    mountError(root, error);
    return;
  }
  const state = { metric: "balanced_top_1", scale: "linear" };
  root.classList.remove("wm-widget");
  root.textContent = "";
  const controls = el("div", { class: "wm-controls" });
  const plot = el("div", { class: "wm-plot", role: "img", "aria-label": "Accuracy against training cost" });
  const table = el("details", { class: "wm-table-view" }, [el("summary", { text: "Table view of the final points" })]);
  const tableBody = el("div", { class: "wm-table-wrap" });
  table.append(tableBody);
  root.append(controls, plot, table);
  controls.append(
    select("Metric", METRICS, state.metric, (v) => { state.metric = v; render(); }),
    select("Cost axis", [["linear", "Linear"], ["log", "Logarithmic"]], state.scale, (v) => { state.scale = v; render(); }),
  );
  const Plotly = await plotly();

  function render() {
    const t = theme();
    const m = state.metric;
    const traces = [];
    const log = state.scale === "log";
    const xmax = Math.max(...Object.values(data.series).flatMap((s) => s.points.map((p) => p.gpu_hours))) * 1.05;
    const xmin = log ? 0.015 : 0;
    for (const key of ["cosine_megadescriptor", "loma_default"]) {
      const flat = data.flats[key];
      if (!flat || flat[m] == null) continue;
      const ours = key === "loma_default";
      traces.push({
        type: "scatter", mode: "lines", name: flat.label + " (no training)", x: [xmin, xmax], y: [flat[m] * 100, flat[m] * 100],
        line: { color: ours ? PALETTE.blue : t.flat, dash: ours ? "dash" : "dot", width: 1.6 },
        hovertemplate: "%{y:.1f} %<extra>" + flat.label + "</extra>",
      });
    }
    for (const key of ["classifier_frozen", "classifier_partial", "classifier_full"]) {
      const s = data.series[key];
      if (!s) continue;
      traces.push({
        type: "scatter", mode: "lines", name: s.label, x: s.points.map((p) => p.gpu_hours), y: s.points.map((p) => p[m] * 100),
        line: { color: t.flat, dash: CLASSIFIER_DASH[key], width: 1.8 },
        customdata: s.points.map((p) => p.epoch),
        hovertemplate: "%{y:.1f} % after %{x:.2f} GPU-h (epoch %{customdata})<extra>" + s.label + "</extra>",
      });
    }
    const ours = data.series.loma_finetuned;
    traces.push({
      type: "scatter", mode: "lines+markers", name: ours.label, x: ours.points.map((p) => p.gpu_hours), y: ours.points.map((p) => p[m] * 100),
      line: { color: PALETTE.blue, width: 2.8 },
      marker: { symbol: "circle", size: 9, color: PALETTE.blue, line: { color: t.ring, width: 1.5 } },
      customdata: ours.points.map((p) => p.epoch),
      hovertemplate: "%{y:.1f} % after %{x:.2f} GPU-h (epoch %{customdata})<extra>" + ours.label + "</extra>",
    });
    const layout = baseLayout(t, {
      hovermode: "closest",
      xaxis: { ...baseLayout(t).xaxis, type: log ? "log" : "linear", title: { text: "training cost (GPU-hours, RTX 4090)", font: { color: t.ink } }, range: log ? [Math.log10(xmin), Math.log10(xmax)] : [-0.2, xmax] },
      yaxis: { ...baseLayout(t).yaxis, title: { text: `${METRICS.find((x) => x[0] === m)[1]} (%)`, font: { color: t.ink } }, rangemode: "tozero" },
      margin: { l: 56, r: 16, t: 64, b: 48 },
    });
    Plotly.react(plot, traces, phoneLayout(layout, { legendItems: traces.length, top: 16 }), PLOT_CONFIG);

    tableBody.textContent = "";
    const rows = [];
    for (const key of ["cosine_megadescriptor", "loma_default"]) {
      const f = data.flats[key];
      if (f) rows.push([f.label, "0", pct(f.top_1), pct(f.top_5), pct(f.balanced_top_1)]);
    }
    for (const key of ["loma_finetuned", "classifier_frozen", "classifier_partial", "classifier_full"]) {
      const s = data.series[key];
      if (!s) continue;
      const last = s.points[s.points.length - 1];
      rows.push([s.label, last.gpu_hours.toFixed(2), pct(last.top_1), pct(last.top_5), pct(last.balanced_top_1)]);
    }
    const head = el("tr", {}, ["Method", "GPU-h", "Top-1", "Top-5", "Bal. top-1"].map((h) => el("th", { text: h })));
    tableBody.append(el("table", { class: "wm-numeric" }, [
      el("thead", {}, head),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el(i ? "td" : "th", { text: c }))))),
    ]));
  }

  render();
  onThemeChange(render);
  onNarrowChange(render);
}
