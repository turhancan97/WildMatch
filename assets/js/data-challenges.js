// Data challenges: confirmed raw-photo examples per challenge category (mount points
// #wm-challenge-<category>) and the image-quality audit summary table
// (#wm-image-quality-table). Data: docs/assets/datasets/challenges/challenges.json and
// docs/data/image_quality_summary.json (scripts/export_data_challenges.py).
import { el, assetUrl, loadJson, pct, mountError } from "./wm-common.js";

let challengesPromise = null;
function loadChallenges() {
  if (!challengesPromise) {
    challengesPromise = fetch(assetUrl("datasets/challenges/challenges.json")).then((r) => {
      if (!r.ok) throw new Error(`challenges.json: HTTP ${r.status}`);
      return r.json();
    });
  }
  return challengesPromise;
}

const MEASURE_LABEL = {
  overexposure: ["saturated_fraction", "saturated", (v) => `${pct(v, 0)} %`],
  empty_frame: ["foreground_fraction", "foreground", (v) => `${pct(v, 1)} %`],
  insufficient_detail: ["foreground_fraction", "foreground", (v) => `${pct(v, 2)} %`],
  corruption: ["std_luma", "contrast σ", (v) => v.toFixed(1)],
  blur: ["sharpness", "sharpness", (v) => v.toFixed(1)],
  night_infrared: ["mean_luma", "mean luma", (v) => v.toFixed(0)],
  occlusion: ["largest_component_fraction", "largest mask part", (v) => `${pct(v, 0)} %`],
};

export async function mountChallenge(root) {
  let data;
  try {
    data = await loadChallenges();
  } catch (error) {
    root.textContent = "The examples appear once they are exported.";
    root.classList.add("wm-widget--pending");
    return;
  }
  const category = root.dataset.category;
  const items = data.items.filter((item) => item.category === category);
  root.classList.remove("wm-widget");
  root.textContent = "";
  if (!items.length) {
    root.append(el("p", { class: "wm-match-caption", text: "No confirmed examples in this category yet." }));
    return;
  }
  const grid = el("div", { class: "wm-gallery" });
  const [key, label, format] = MEASURE_LABEL[category] || ["foreground_fraction", "foreground", (v) => pct(v)];
  for (const item of items) {
    const src = assetUrl(`datasets/challenges/${item.image.file}`);
    const parts = [`${item.dataset_label} · ${item.side}`];
    if (item.measurements[key] != null) parts.push(`${label} ${format(item.measurements[key])}`);
    if (item.sam3_n_instances != null && category === "occlusion") parts.push(`${item.sam3_n_instances} mask parts`);
    if (item.query_top1_rate != null && item.query_runs) parts.push(`Top-1 ${pct(item.query_top1_rate, 0)} % over ${item.query_runs} runs`);
    const figure = el("figure", { class: "wm-gallery-item" }, [
      el("a", { href: src, target: "_blank", rel: "noopener", title: item.note }, [
        el("img", { src, alt: `${item.dataset_label}: ${item.note}`, loading: "lazy", width: String(item.image.width), height: String(item.image.height) }),
      ]),
      el("figcaption", { text: parts.join(" · ") }),
    ]);
    grid.append(figure);
  }
  root.append(grid);
  root.append(el("p", { class: "wm-match-caption", text: `${items.length} examples confirmed by eye; captions give the audit measurement that ranked the image and, for queries, the Top-1 rate over completed runs. Click a photo for the full-size copy.` }));
}

const FLAG_COLUMNS = [
  ["overexposed_images", "Over-exposed"], ["underexposed_images", "Under-exposed"], ["low_contrast_images", "Low contrast"],
  ["tiny_foreground_images", "Tiny animal"], ["empty_foreground_images", "Empty"], ["fragmented_mask_images", "Fragmented mask"],
  ["blurry_images", "Blurry (2 %)"],
];

export async function mountImageQualityTable(root) {
  let data;
  try {
    data = await loadJson("image_quality_summary.json");
  } catch (error) {
    mountError(root, error);
    return;
  }
  root.classList.remove("wm-widget");
  root.textContent = "";
  const head = ["Dataset", "Images", "Flagged", ...FLAG_COLUMNS.map((c) => c[1]), "Flagged queries", "Top-1 flagged", "Top-1 clean", "Runs"];
  const table = el("table", { class: "wm-results wm-ss-table wm-iq-table" }, [el("thead", {}, el("tr", {}, head.map((h) => el("th", { text: h }))))]);
  const body = el("tbody");
  for (const r of data.rows) {
    const cells = [
      r.label, r.images.toLocaleString(), `${r.any_flag_images.toLocaleString()} (${pct(r.any_flag_fraction, 1)} %)`,
      ...FLAG_COLUMNS.map((c) => (r[c[0]] == null ? "–" : r[c[0]].toLocaleString())),
      r.flagged_queries == null ? "–" : r.flagged_queries.toLocaleString(),
      r.query_top1_flagged == null ? "–" : `${pct(r.query_top1_flagged, 0)} %`,
      r.query_top1_clean == null ? "–" : `${pct(r.query_top1_clean, 0)} %`,
      r.scored_runs == null ? "–" : String(r.scored_runs),
    ];
    body.append(el("tr", {}, cells.map((c, i) => el(i === 0 ? "th" : "td", { text: c }))));
  }
  table.append(body);
  root.append(el("div", { class: "wm-table-wrap" }, [table]));
  const rules = Object.entries(data.flag_rules).map(([k, v]) => `${k}: ${v}`).join("; ");
  root.append(el("p", { class: "wm-match-caption", text: `Flags (rules: ${rules}). "Flagged" excludes the blur ranking. Top-1 columns average each query's Top-1 over all completed runs of all methods; flagged Hyena queries score as well as clean ones, while flagged Sea star and Turtle queries score far lower (Nyala has only two flagged queries). CzechLynx is audited once for both splits.` }));
}
