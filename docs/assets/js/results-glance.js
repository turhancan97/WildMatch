// At-a-glance tiles at the top of the Results page, computed from docs/data/results.json
// at the paper's main budget: datasets with a Top-5 gain, the median Top-5 gain, the median
// balanced Top-1 gain, and the training cost of the CzechLynx matcher from adapt.json.
import { loadJson, el, mountError } from "./wm-common.js";

const PAPER_DATASETS = ["czechlynx", "hyena", "leopard", "nyala", "salamander", "sea_star", "whale_shark", "turtle"];

function median(values) {
  const v = [...values].sort((a, b) => a - b);
  if (!v.length) return null;
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

function points(value, digits = 1) {
  const p = value * 100;
  return `${p >= 0 ? "+" : ""}${p.toFixed(digits)}`;
}

export async function mountResultsGlance(root) {
  let results;
  let adapt = null;
  try {
    results = await loadJson("results.json");
    try { adapt = await loadJson("adapt.json"); } catch (error) { adapt = null; }
  } catch (error) {
    mountError(root, error);
    return;
  }
  const k = results.main_k;
  const rows = results.rows.filter((r) => r.k === k && PAPER_DATASETS.includes(r.dataset));
  const byDataset = {};
  for (const r of rows) {
    if (r.method_key !== "loma_finetuned" && r.method_key !== "loma_default") continue;
    byDataset[r.dataset] = byDataset[r.dataset] || {};
    byDataset[r.dataset][r.method_key] = r;
  }
  const gains5 = [];
  const gainsB = [];
  for (const d of PAPER_DATASETS) {
    const pair = byDataset[d];
    if (!pair || !pair.loma_finetuned || !pair.loma_default) continue;
    gains5.push(pair.loma_finetuned.top_5 - pair.loma_default.top_5);
    gainsB.push(pair.loma_finetuned.balanced_top_1 - pair.loma_default.balanced_top_1);
  }
  const improved5 = gains5.filter((g) => g > 0).length;
  const improvedB = gainsB.filter((g) => g > 0).length;
  let gpuHours = null;
  if (adapt && Array.isArray(adapt.rows)) {
    const row = adapt.rows.find((r) => (r.matcher || "").toLowerCase() === "loma" && r.variant === "matcher");
    if (row && row.gpu_hours != null) gpuHours = row.gpu_hours;
  }
  root.classList.remove("wm-widget");
  root.textContent = "";
  const tiles = el("div", { class: "wm-kpis wm-kpis--glance" });
  const tile = (label, value, note, cls) => el("div", { class: `wm-kpi ${cls || ""}` }, [
    el("span", { class: "wm-kpi-label", text: label }),
    el("span", { class: "wm-kpi-value", text: value }),
    note ? el("span", { class: "wm-kpi-note", text: note }) : null,
  ]);
  tiles.append(
    tile("Datasets with higher Top-5", `${improved5} of ${gains5.length}`, `LoMa + WildMatch over the default matcher at k = ${k}`, "wm-kpi--ours"),
    tile("Median Top-5 gain", `${points(median(gains5))} pts`, `range ${points(Math.min(...gains5))} to ${points(Math.max(...gains5))}`, "wm-kpi--ours"),
    tile("Median balanced Top-1 gain", `${points(median(gainsB))} pts`, `${improvedB} of ${gainsB.length} datasets improve`, "wm-kpi--ours"),
    tile("Training cost", gpuHours != null ? `${Number(gpuHours).toFixed(1)} GPU-h` : "about 5 GPU-h", "CzechLynx, matching module only, RTX 4090; identity labels only"),
  );
  root.append(tiles);
}
