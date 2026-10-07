// Rare vs common individuals: accuracy by the number of gallery images of the query's
// individual, default LoMa beside LoMa + WildMatch, with the shortlist ceiling and a
// bootstrap interval on the gain. Data: docs/data/frequency_bins.json
// (scripts/export_frequency_bins.py, from the paper's k=250 runs).
import { PALETTE, theme, onThemeChange, loadJson, plotly, pct, el, select, baseLayout, PLOT_CONFIG, mountError, isNarrow, onNarrowChange, phoneLayout } from "./wm-common.js";

const METRICS = [["top_5", "Top-5"], ["top_1", "Top-1"]];

function signed(value, digits = 1) {
  if (value == null || Number.isNaN(value)) return "–";
  const v = value * 100;
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;
}

export async function mountFrequencyBins(root) {
  let data;
  try {
    data = await loadJson("frequency_bins.json");
  } catch (error) {
    mountError(root, error);
    return;
  }
  const Plotly = await plotly();
  const state = { dataset: data.datasets[0].key, metric: "top_5" };
  root.classList.remove("wm-widget");
  root.textContent = "";

  const controls = el("div", { class: "wm-controls" });
  const plot = el("div", { class: "wm-plot wm-plot--short", role: "img", "aria-label": "Accuracy by number of gallery images per individual" });
  const tableWrap = el("div", { class: "wm-table-wrap" });
  const caption = el("p", { class: "wm-match-caption" });
  root.append(controls, plot, tableWrap, caption);
  controls.append(
    select("Dataset", data.datasets.map((d) => [d.key, d.label]), state.dataset, (v) => { state.dataset = v; render(); }),
    select("Metric", METRICS, state.metric, (v) => { state.metric = v; render(); }),
  );

  function current() { return data.datasets.find((d) => d.key === state.dataset); }

  function render() {
    const t = theme();
    const d = current();
    const m = state.metric;
    const metricLabel = METRICS.find((x) => x[0] === m)[1];
    const bins = d.bins;
    // Phones show only the bin label on the axis (the long labels ran off the right edge);
    // the query counts stay in the table below.
    const x = bins.map((b) => (isNarrow() ? b.label : `${b.label}\n(${b.n.toLocaleString()} queries)`));
    const small = bins.map((b) => b.small || b.n === 0);
    const val = (b, who) => (b.n ? b[m][who] * 100 : null);
    const traces = [
      { type: "bar", name: "Default LoMa", x, y: bins.map((b) => val(b, "default")), offsetgroup: 0,
        marker: { color: PALETTE.blue, opacity: small.map((s) => (s ? 0.25 : 0.45)), pattern: { shape: "/", fgcolor: PALETTE.blue, bgcolor: "rgba(0,0,0,0)", size: 6, solidity: 0.35 },
          line: { color: PALETTE.blue, width: 1.5 } },
        hovertemplate: `%{x}<br>${metricLabel} default %{y:.1f} %<extra></extra>` },
      { type: "bar", name: "LoMa + WildMatch", x, y: bins.map((b) => val(b, "finetuned")), offsetgroup: 1,
        marker: { color: PALETTE.blue, opacity: small.map((s) => (s ? 0.45 : 1)), line: { color: PALETTE.blue, width: 1.5 } },
        hovertemplate: `%{x}<br>${metricLabel} fine-tuned %{y:.1f} %<extra></extra>` },
      { type: "scatter", mode: "markers", name: "individual inside the shortlist (ceiling)", x,
        y: bins.map((b) => (b.n ? b.shortlist_share * 100 : null)),
        marker: { symbol: "diamond", size: 10, color: t.flat, line: { color: t.ring, width: 1 } },
        hovertemplate: "%{x}<br>shortlist share %{y:.1f} %<extra></extra>" },
    ];
    const layout = baseLayout(t, {
      barmode: "group", bargap: 0.25, bargroupgap: 0.05, hovermode: "closest",
      margin: { l: 56, r: 16, t: 40, b: 64 },
      xaxis: { ...baseLayout(t).xaxis, title: { text: "gallery images of the query's individual", font: { color: t.ink } }, tickfont: { color: t.muted, size: 11 } },
      yaxis: { ...baseLayout(t).yaxis, title: { text: `${metricLabel} (%)`, font: { color: t.ink } }, range: [0, 104] },
      annotations: bins.map((b, i) => (b.n ? {
        x: x[i], y: Math.max(val(b, "default") || 0, val(b, "finetuned") || 0) + 3, text: signed(b[m].gain), showarrow: false,
        font: { color: b[m].gain >= 0 ? PALETTE.blue : PALETTE.red, size: 11 }, yanchor: "bottom",
      } : null)).filter(Boolean),
    });
    Plotly.react(plot, traces, phoneLayout(layout, { legendItems: traces.length, top: 16 }), PLOT_CONFIG);

    const head = ["Gallery images", "Queries", "Individuals", "In shortlist", `${metricLabel} default`, `${metricLabel} fine-tuned`, "Gain (95 % interval)"];
    const table = el("table", { class: "wm-results wm-ss-table" }, [el("thead", {}, el("tr", {}, head.map((h) => el("th", { text: h }))))]);
    const body = el("tbody");
    const row = (label, b, ids, cls) => {
      const cells = b.n
        ? [label, b.n.toLocaleString(), ids, `${pct(b.shortlist_share)} %`, `${pct(b[m].default)} %`, `${pct(b[m].finetuned)} %`,
           `${signed(b[m].gain)} [${signed(b[m].gain_ci95[0])}, ${signed(b[m].gain_ci95[1])}]${b.small ? " · small bin" : ""}`]
        : [label, "0", "0", "–", "–", "–", "–"];
      return el("tr", { class: cls || "" }, cells.map((c, i) => el(i === 0 ? "th" : "td", { text: c })));
    };
    for (const b of bins) body.append(row(b.label, b, b.n_identities.toLocaleString(), b.small ? "wm-fb-small" : ""));
    body.append(row("all with a gallery image", d.overall, d.n_identities_database.toLocaleString(), "wm-ss-ours"));
    table.append(body);
    tableWrap.textContent = "";
    tableWrap.append(table);
    caption.textContent = `${d.label}: ${d.n_queries.toLocaleString()} queries against ${d.n_database.toLocaleString()} gallery images of ${d.n_identities_database.toLocaleString()} individuals; k = ${d.k}. `
      + `Numbers above the bars are the fine-tuning gain in points; hatched bars are the default matcher, solid bars the fine-tuned one, diamonds the shortlist ceiling. `
      + (d.n_queries_without_gallery_image ? `${d.n_queries_without_gallery_image.toLocaleString()} queries whose individual has no gallery image are excluded. ` : "")
      + `Bins with fewer than ${data.small_bin} queries are drawn faint.`;
  }

  onThemeChange(render);
  onNarrowChange(render);
  render();
}
