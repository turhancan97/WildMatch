// Candidate-budget trade-off: a slider over the measured budgets k with readouts for the
// shortlist share, Top-5 of both matchers and the matching time, plus a two-panel chart
// with the chosen k marked. Data: docs/data/budget_tradeoff.json
// (scripts/export_budget_tradeoff.py; every value is a measurement, nothing is interpolated).
import { PALETTE, SERIES_STYLE, theme, onThemeChange, loadJson, plotly, pct, el, select, baseLayout, PLOT_CONFIG, mountError } from "./wm-common.js";

function fmtMinutes(value) {
  if (value == null) return "–";
  if (value < 1) return `${(value * 60).toFixed(0)} s`;
  if (value < 10) return `${value.toFixed(1)} min`;
  return `${value.toFixed(0)} min`;
}

export async function mountBudgetTradeoff(root) {
  let data;
  try {
    data = await loadJson("budget_tradeoff.json");
  } catch (error) {
    mountError(root, error);
    return;
  }
  const Plotly = await plotly();
  const state = { dataset: data.datasets[0].key, index: data.budgets.indexOf(data.main_k) };
  root.classList.remove("wm-widget");
  root.textContent = "";

  const controls = el("div", { class: "wm-controls" });
  const tiles = el("div", { class: "wm-kpis" });
  const plot = el("div", { class: "wm-plot wm-plot--short", role: "img", "aria-label": "Shortlist share, Top-5 and matching time against the candidate budget" });
  const caption = el("p", { class: "wm-match-caption" });
  root.append(controls, tiles, plot, caption);

  const slider = el("input", { type: "range", min: "0", max: String(data.budgets.length - 1), step: "1", value: String(state.index), "aria-label": "candidate budget k" });
  const sliderValue = el("span", { class: "wm-range-value wm-kpi-k", text: `k = ${data.budgets[state.index]}` });
  slider.addEventListener("input", () => { state.index = Number(slider.value); render(); });
  const ticks = el("div", { class: "wm-range-ticks" }, data.budgets.map((k) => el("span", { text: String(k) })));
  controls.append(
    select("Dataset", data.datasets.map((d) => [d.key, d.label]), state.dataset, (v) => { state.dataset = v; render(); }),
    el("label", { class: "wm-control wm-control--wide" }, [
      el("span", { text: "Candidate budget k (measured budgets only)" }),
      el("span", { class: "wm-range wm-range--wide" }, [slider, sliderValue]),
      ticks,
    ]),
  );

  function current() { return data.datasets.find((d) => d.key === state.dataset); }

  function tile(label, value, note, cls) {
    return el("div", { class: `wm-kpi ${cls || ""}` }, [
      el("span", { class: "wm-kpi-label", text: label }),
      el("span", { class: "wm-kpi-value", text: value }),
      note ? el("span", { class: "wm-kpi-note", text: note }) : null,
    ]);
  }

  function render() {
    const t = theme();
    const d = current();
    const k = data.budgets[state.index];
    const b = d.budgets[state.index];
    sliderValue.textContent = `k = ${k}`;
    const gain = (b.top_5.finetuned - b.top_5.default) * 100;
    tiles.textContent = "";
    tiles.append(
      tile("Individual inside the shortlist", `${pct(b.shortlist_share)} %`, "of queries; the ceiling for any matcher ranking these candidates"),
      tile("Top-5, default LoMa", `${pct(b.top_5.default)} %`, null, "wm-kpi--default"),
      tile("Top-5, LoMa + WildMatch", `${pct(b.top_5.finetuned)} %`, `${gain >= 0 ? "+" : ""}${gain.toFixed(1)} points over default`, "wm-kpi--ours"),
      tile("Matching time per 1,000 queries", fmtMinutes(b.display.minutes_per_1000_queries), `${b.display.ms_per_pair.toFixed(2)} ms per pair · ${b.display.gpu}`),
    );

    const ks = d.budgets.map((x) => x.k);
    const share = d.budgets.map((x) => x.shortlist_share * 100);
    const top5d = d.budgets.map((x) => x.top_5.default * 100);
    const top5f = d.budgets.map((x) => x.top_5.finetuned * 100);
    const minutes = d.budgets.map((x) => x.display.minutes_per_1000_queries);
    const hw = d.budgets.map((x) => x.display.gpu);
    const sf = SERIES_STYLE.loma_finetuned;
    const sd = SERIES_STYLE.loma_default;
    const traces = [
      { type: "scatter", mode: "lines+markers", x: ks, y: share, name: "individual inside the shortlist", xaxis: "x", yaxis: "y",
        line: { color: t.flat, width: 2, dash: "dot" }, marker: { symbol: "diamond", size: 7, color: t.flat },
        hovertemplate: "k = %{x}<br>shortlist share %{y:.1f} %<extra></extra>" },
      { type: "scatter", mode: "lines+markers", x: ks, y: top5d, name: "Top-5, default LoMa", xaxis: "x", yaxis: "y",
        line: { color: sd.color, width: sd.width, dash: sd.dash }, marker: { symbol: sd.symbol, size: 8, color: sd.color, line: { color: sd.color, width: 1.5 } },
        hovertemplate: "k = %{x}<br>Top-5 default %{y:.1f} %<extra></extra>" },
      { type: "scatter", mode: "lines+markers", x: ks, y: top5f, name: "Top-5, LoMa + WildMatch", xaxis: "x", yaxis: "y",
        line: { color: sf.color, width: sf.width, dash: sf.dash }, marker: { symbol: sf.symbol, size: 8, color: sf.color },
        hovertemplate: "k = %{x}<br>Top-5 fine-tuned %{y:.1f} %<extra></extra>" },
      { type: "scatter", mode: "lines+markers", x: ks, y: minutes, name: "matching minutes per 1,000 queries", xaxis: "x2", yaxis: "y2",
        line: { color: PALETTE.gold, width: 2.4 }, marker: { symbol: "triangle-up", size: 8, color: PALETTE.gold }, customdata: hw,
        hovertemplate: "k = %{x}<br>%{y:.1f} min per 1,000 queries<br>%{customdata}<extra></extra>" },
    ];
    const base = baseLayout(t);
    const logAxis = (domain, title) => ({ ...base.xaxis, type: "log", domain, tickvals: ks, ticktext: ks.map(String),
      title: { text: title, font: { color: t.ink } } });
    const layout = {
      ...base,
      hovermode: "closest",
      margin: { l: 56, r: 16, t: 56, b: 52 },
      legend: { ...base.legend, y: 1.18 },
      xaxis: logAxis([0, 0.56], "candidate budget k"),
      xaxis2: logAxis([0.64, 1], "candidate budget k"),
      yaxis: { ...base.yaxis, title: { text: "% of queries", font: { color: t.ink } }, range: [0, 102], anchor: "x" },
      yaxis2: { ...base.yaxis, title: { text: "minutes per 1,000 queries", font: { color: t.ink } }, rangemode: "tozero", anchor: "x2" },
      shapes: [0, 1].map((i) => ({ type: "line", xref: i === 0 ? "x" : "x2", yref: "paper", x0: k, x1: k, y0: 0, y1: 1,
        line: { color: PALETTE.red, width: 1.5, dash: "dash" } })),
      annotations: [
        { text: "<b>Accuracy</b>", showarrow: false, xref: "paper", yref: "paper", x: 0.0, y: 1.06, xanchor: "left", font: { color: t.ink, size: 13 } },
        { text: "<b>Matching cost</b>", showarrow: false, xref: "paper", yref: "paper", x: 0.64, y: 1.06, xanchor: "left", font: { color: t.ink, size: 13 } },
      ],
    };
    Plotly.react(plot, traces, layout, PLOT_CONFIG);
    const unknown = d.runs_total - d.runs_with_recorded_hardware;
    caption.textContent = `${d.label}: ${d.n_queries.toLocaleString()} queries; at k = ${k} each matcher scores ${(b.display.pairs).toLocaleString()} pairs. `
      + `Times are the Vismatch matching timer only (feature extraction excluded); hardware from Slurm accounting, ${d.dominant_gpu} for most runs`
      + (unknown ? `, not recorded for ${unknown} of ${d.runs_total} runs` : "") + ". The dashed red line marks the chosen k.";
  }

  onThemeChange(render);
  render();
}
