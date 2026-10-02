// Sortable, filterable results table over docs/data/results.json with CSV download.
import { loadJson, pct, el, select, mountError, SERIES_STYLE } from "./wm-common.js";

const COLUMNS = [
  ["dataset_label", "Dataset", "text"],
  ["method_label", "Method", "text"],
  ["backbone", "Backbone", "text"],
  ["k", "k", "num"],
  ["top_1", "Top-1", "pct"],
  ["top_5", "Top-5", "pct"],
  ["top_10", "Top-10", "pct"],
  ["balanced_top_1", "Bal. top-1", "pct"],
  ["mAP_at_k", "mAP@k", "pct"],
  ["runtime_min", "Compute (min)", "num"],
];

export async function mountResultsTable(root) {
  let data;
  try {
    data = await loadJson("results.json");
  } catch (error) {
    mountError(root, error);
    return;
  }
  const labels = Object.fromEntries(data.datasets.map((d) => [d.key, d.label]));
  const rows = data.rows.map((r) => ({ ...r, dataset_label: labels[r.dataset] || r.dataset }));
  const state = { dataset: "all", k: String(data.main_k), family: "all", sort: "dataset_label", dir: 1 };

  root.classList.remove("wm-widget");
  root.textContent = "";
  const controls = el("div", { class: "wm-controls" });
  const wrap = el("div", { class: "wm-table-wrap" });
  const foot = el("p", { class: "wm-table-foot" });
  root.append(controls, wrap, foot);

  const ks = [...new Set(rows.map((r) => r.k).filter((k) => k != null))].sort((a, b) => a - b);
  controls.append(
    select("Dataset", [["all", "All datasets"], ...data.datasets.map((d) => [d.key, d.label])], state.dataset, (v) => { state.dataset = v; render(); }),
    select("Candidate budget", [["all", "All budgets"], ...ks.map((k) => [String(k), `k = ${k}`])], state.k, (v) => { state.k = v; render(); }),
    select("Method group", [["all", "All methods"], ["ours", "Fine-tuned matchers (ours)"], ["matchers", "All matchers"], ["baseline", "Baselines"]], state.family, (v) => { state.family = v; render(); }),
    el("button", { class: "md-button wm-button", type: "button", onclick: download, text: "Download CSV" }),
  );

  function filtered() {
    return rows.filter((r) => {
      if (state.dataset !== "all" && r.dataset !== state.dataset) return false;
      // Budget-independent baselines (k = null) are shown with every budget.
      if (state.k !== "all" && r.k != null && String(r.k) !== state.k) return false;
      if (state.family === "ours" && !r.method_key.endsWith("finetuned")) return false;
      if (state.family === "matchers" && r.family === "baseline") return false;
      if (state.family === "baseline" && r.family !== "baseline") return false;
      return true;
    });
  }

  function sorted(list) {
    const [key, dir] = [state.sort, state.dir];
    const kind = COLUMNS.find((c) => c[0] === key)?.[2] || "text";
    return [...list].sort((a, b) => {
      const va = a[key], vb = b[key];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return (kind === "text" ? String(va).localeCompare(String(vb)) : va - vb) * dir;
    });
  }

  function cell(row, [key, , kind]) {
    const v = row[key];
    if (kind === "pct") return pct(v);
    if (key === "k") return v == null ? "–" : String(v);
    if (kind === "num") return v == null ? "–" : Number(v).toFixed(2);
    return v == null ? "–" : String(v);
  }

  function render() {
    const list = sorted(filtered());
    const head = el("tr", {}, COLUMNS.map(([key, label]) => {
      const active = state.sort === key;
      return el("th", {
        class: `wm-sortable${active ? (state.dir > 0 ? " is-asc" : " is-desc") : ""}`,
        "aria-sort": active ? (state.dir > 0 ? "ascending" : "descending") : "none",
        onclick: () => { if (state.sort === key) state.dir *= -1; else { state.sort = key; state.dir = 1; } render(); },
        text: label, scope: "col",
      });
    }));
    const body = list.map((row) => {
      const tr = el("tr", { class: row.method_key.endsWith("finetuned") ? "wm-ours" : "" }, COLUMNS.map((c, i) => {
        const td = el(i === 1 ? "th" : "td", { text: cell(row, c) });
        if (i === 1) {
          td.setAttribute("scope", "row");
          const color = SERIES_STYLE[row.method_key]?.color;
          if (color) td.prepend(el("span", { class: "wm-swatch", style: `--wm-swatch:${color}` }));
        }
        return td;
      }));
      return tr;
    });
    wrap.textContent = "";
    wrap.append(el("table", { class: "wm-numeric wm-results" }, [el("thead", {}, head), el("tbody", {}, body)]));
    foot.textContent = `${list.length} rows. Values in percent; "–" marks a budget-independent method or a metric the run does not report. mAP@k is the shortlist-aware mAP at the candidate budget.`;
  }

  function download() {
    const list = sorted(filtered());
    const header = COLUMNS.map((c) => c[1]).concat(["run_id"]);
    const lines = [header.join(",")].concat(list.map((r) => COLUMNS.map((c) => {
      const v = r[c[0]];
      return v == null ? "" : String(v).includes(",") ? `"${v}"` : String(v);
    }).concat([r.run_id || ""]).join(",")));
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
    const a = el("a", { href: URL.createObjectURL(blob), download: "wildmatch_results.csv" });
    document.body.append(a); a.click(); a.remove();
  }

  render();
}
