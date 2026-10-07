// Score separation: histograms of LoMa image scores for same-individual and
// different-individual shortlist pairs, default matcher beside LoMa + WildMatch, with a
// per-query margin view and a statistics table. Data:
// docs/assets/demo/score_separation/score_separation.json
// (scripts/export_score_separation.py, from the paper's k=250 runs' scores.npz).
import { PALETTE, theme, onThemeChange, el, assetUrl, select, checkbox, plotly, baseLayout, PLOT_CONFIG, isNarrow, onNarrowChange, phoneLayout } from "./wm-common.js";

const VIEWS = [["pairs", "Pair scores: same vs different individual"], ["margin", "Per-query margin: best same − best different"]];
const SAME = PALETTE.blue;
const DIFF = PALETTE.red;

function fmt(value, digits = 3) {
  return value == null || Number.isNaN(value) ? "–" : Number(value).toFixed(digits);
}

function midpoints(edges) {
  const mids = [];
  for (let i = 0; i + 1 < edges.length; i += 1) mids.push((edges[i] + edges[i + 1]) / 2);
  return mids;
}

export async function mountScoreSeparation(root) {
  let data;
  try {
    const response = await fetch(assetUrl("demo/score_separation/score_separation.json"));
    if (!response.ok) throw new Error(`score_separation.json: HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    root.textContent = "The score-separation view appears once its histograms are exported.";
    root.classList.add("wm-widget--pending");
    return;
  }
  const Plotly = await plotly();
  const state = { dataset: data.datasets[0].key, view: "pairs", log: false, share: true };
  root.classList.remove("wm-widget");
  root.textContent = "";

  const controls = el("div", { class: "wm-controls" });
  const plot = el("div", { class: "wm-plot wm-plot--short" });
  const statsWrap = el("div", { class: "wm-table-wrap" });
  const caption = el("p", { class: "wm-match-caption" });
  root.append(controls, plot, statsWrap, caption);

  controls.append(
    select("Dataset", data.datasets.map((d) => [d.key, d.label]), state.dataset, (v) => { state.dataset = v; render(); }),
    select("View", VIEWS, state.view, (v) => { state.view = v; render(); }),
    el("div", { class: "wm-control" }, [
      el("span", { text: "Axes" }),
      el("div", { class: "wm-check-group" }, [
        checkbox("share of pairs instead of counts", state.share, (v) => { state.share = v; render(); }),
        checkbox("log y-axis", state.log, (v) => { state.log = v; render(); }),
      ]),
    ]),
  );

  function current() {
    return data.datasets.find((d) => d.key === state.dataset);
  }

  function barTrace(x, counts, name, color, xaxis, yaxis, width, showlegend) {
    const total = counts.reduce((a, b) => a + b, 0) || 1;
    const y = state.share ? counts.map((c) => c / total) : counts;
    return {
      type: "bar", x, y, name, xaxis, yaxis, width, showlegend, legendgroup: name,
      marker: { color, line: { color, width: 0.5 } }, opacity: 0.62,
      customdata: counts.map((c) => [c, c / total]),
      hovertemplate: `${name}<br>%{customdata[0]:,} pairs (%{customdata[1]:.2%})<extra></extra>`,
    };
  }

  function render() {
    const t = theme();
    const d = current();
    const isMargin = state.view === "margin";
    const edges = isMargin ? data.margin_bin_edges : data.bin_edges;
    const x = midpoints(edges);
    const width = edges[1] - edges[0];
    const traces = [];
    data.matchers.forEach((m, index) => {
      const ax = index === 0 ? "" : "2";
      const side = d.matchers[m.key];
      if (isMargin) {
        const counts = side.margin;
        const total = counts.reduce((a, b) => a + b, 0) || 1;
        traces.push({
          type: "bar", x, y: state.share ? counts.map((c) => c / total) : counts, width,
          name: "per-query margin", xaxis: `x${ax}`, yaxis: `y${ax}`, showlegend: index === 0, legendgroup: "margin",
          marker: { color: x.map((v) => (v >= 0 ? SAME : DIFF)) }, opacity: 0.75,
          customdata: counts.map((c) => [c, c / total]),
          hovertemplate: "margin %{x:.2f}<br>%{customdata[0]:,} queries (%{customdata[1]:.2%})<extra></extra>",
        });
      } else {
        traces.push(barTrace(x, side.same, "same individual", SAME, `x${ax}`, `y${ax}`, width, index === 0));
        traces.push(barTrace(x, side.different, "different individual", DIFF, `x${ax}`, `y${ax}`, width, index === 0));
      }
    });
    const base = baseLayout(t);
    const axisX = (title) => ({ ...base.xaxis, title: { text: title, font: { color: t.ink } }, range: isMargin ? [-1, 1] : [0, 1] });
    const axisY = (title, anchor) => ({ ...base.yaxis, type: state.log ? "log" : "linear", anchor,
      title: title ? { text: title, font: { color: t.ink } } : undefined, tickformat: state.share && !state.log ? ".0%" : undefined });
    const labels = data.matchers.map((m) => m.label);
    // Phones stack the two matchers' panels; side by side their titles collide at 390 px.
    const narrow = isNarrow();
    const xTitle = isMargin ? "best same − best different score" : "image score";
    const yTitle = state.share ? (isMargin ? "share of queries" : "share of pairs") : (isMargin ? "queries" : "pairs");
    const layout = narrow ? phoneLayout({
      ...base,
      barmode: "overlay",
      bargap: 0.05,
      hovermode: "x",
      margin: { l: 60, r: 16, t: 28, b: 52 },
      xaxis: { ...axisX(xTitle), domain: [0, 1], anchor: "y" },
      xaxis2: { ...axisX(xTitle), domain: [0, 1], anchor: "y2" },
      yaxis: { ...axisY(yTitle, "x"), domain: [0.58, 1] },
      yaxis2: { ...axisY(yTitle, "x2"), domain: [0, 0.38], matches: "y" },
      annotations: labels.map((label, i) => ({
        text: `<b>${label}</b>`, showarrow: false, xref: "paper", yref: "paper", x: 0, xanchor: "left", y: i === 0 ? 1.01 : 0.39, yanchor: "bottom",
        font: { color: t.ink, size: 13 },
      })),
      shapes: isMargin ? [["x", "y"], ["x2", "y2"]].map(([xr, yr]) => ({
        type: "line", xref: xr, yref: `${yr} domain`, x0: 0, x1: 0, y0: 0, y1: 1,
        line: { color: t.muted, width: 1, dash: "dot" },
      })) : [],
    }, { legendItems: isMargin ? 1 : 2, height: 620 }) : {
      ...base,
      barmode: "overlay",
      bargap: 0.05,
      hovermode: "x",
      margin: { l: 60, r: 16, t: 48, b: 52 },
      xaxis: { ...axisX(isMargin ? "best same − best different score" : "image score"), domain: [0, 0.48] },
      xaxis2: { ...axisX(isMargin ? "best same − best different score" : "image score"), domain: [0.52, 1] },
      yaxis: axisY(state.share ? (isMargin ? "share of queries" : "share of pairs") : (isMargin ? "queries" : "pairs"), "x"),
      yaxis2: { ...axisY(null, "x2"), matches: "y" },
      annotations: labels.map((label, i) => ({
        text: `<b>${label}</b>`, showarrow: false, xref: "paper", yref: "paper", x: i === 0 ? 0.24 : 0.76, y: 1.07,
        font: { color: t.ink, size: 13 },
      })),
      shapes: isMargin ? [0, 1].map((i) => ({
        type: "line", xref: i === 0 ? "x" : "x2", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1,
        line: { color: t.muted, width: 1, dash: "dot" },
      })) : [],
    };
    Plotly.react(plot, traces, layout, PLOT_CONFIG);
    renderStats(d, isMargin);
    caption.textContent = isMargin
      ? `${d.label}: ${d.n_queries_with_same_candidate.toLocaleString()} of ${d.n_queries.toLocaleString()} queries have a same-individual candidate in their ${d.k}-image shortlist; the margin is defined for them. Bars right of zero are queries whose own individual outscores every other candidate.`
      : `${d.label}: ${d.runs.default.n_pairs.toLocaleString()} shortlist pairs from ${d.n_queries.toLocaleString()} queries × ${d.k} MegaDescriptor-L candidates, scored by both matchers. Different-individual pairs are hard shortlist candidates, not random pairs.`;
  }

  function renderStats(d, isMargin) {
    const head = isMargin
      ? ["Matcher", "Queries with margin", "Median margin", "Positive margin", "Recorded Top-1"]
      : ["Matcher", "AUROC same > different", "Median same", "Median different", "Histogram overlap", "Same pairs", "Different pairs"];
    const table = el("table", { class: "wm-results wm-ss-table" }, [
      el("thead", {}, el("tr", {}, head.map((h) => el("th", { text: h })))),
    ]);
    const body = el("tbody");
    for (const m of data.matchers) {
      const s = d.matchers[m.key].stats;
      const cells = isMargin
        ? [m.label, s.n_queries_with_margin.toLocaleString(), fmt(s.median_margin), fmt(s.positive_margin_fraction * 100, 1) + " %", fmt(d.runs[m.key].top_1 * 100, 1) + " %"]
        : [m.label, fmt(s.auroc), fmt(s.median_same), fmt(s.median_different), fmt(s.overlap), s.n_same.toLocaleString(), s.n_different.toLocaleString()];
      body.append(el("tr", { class: m.key === "finetuned" ? "wm-ss-ours" : "" }, cells.map((c, i) => el(i === 0 ? "th" : "td", { text: c }))));
    }
    table.append(body);
    statsWrap.textContent = "";
    statsWrap.append(table);
  }

  onThemeChange(render);
  onNarrowChange(render);
  render();
}
