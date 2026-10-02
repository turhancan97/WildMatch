// Rank-change explorer: one query, its top 5 under cosine retrieval, default LoMa and
// LoMa + WildMatch, with the true individual highlighted; filters for queries that
// fine-tuning rescued, still gets wrong, regressed, or already had right. Data:
// docs/assets/demo/rank_change/rank_change.json (scripts/export_rank_change_demo.py).
import { el, assetUrl } from "./wm-common.js";

const FILTERS = [
  ["all", "All examples", "every sampled query"],
  ["rescued", "Rescued", "default LoMa ranked another individual first; the fine-tuned matcher ranks the true one first"],
  ["still_wrong", "Still wrong", "the fine-tuned matcher still ranks another individual first"],
  ["regressed", "Regressed", "default LoMa was right at rank 1 and the fine-tuned matcher is not"],
  ["already_right", "Already right", "both matchers rank the true individual first"],
];

export async function mountRankChange(root) {
  let data;
  try {
    const response = await fetch(assetUrl("demo/rank_change/rank_change.json"));
    if (!response.ok) throw new Error(`rank_change.json: HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    root.textContent = "The rank-change explorer appears once its data is exported.";
    root.classList.add("wm-widget--pending");
    return;
  }
  const state = { filter: "rescued", index: 0 };
  root.classList.remove("wm-widget");
  root.textContent = "";

  const filterBar = el("div", { class: "wm-demo-groups", role: "tablist", "aria-label": "query categories" });
  const countLine = el("p", { class: "wm-rc-counts" });
  const strip = el("div", { class: "wm-demo-strip", role: "list", "aria-label": "queries" });
  const columns = el("div", { class: "wm-rc-columns" });
  const caption = el("p", { class: "wm-match-caption" });
  root.append(filterBar, countLine, el("div", { class: "wm-demo-row" }, [el("span", { class: "wm-demo-label", text: "Query" }), strip]), columns, caption);

  const visible = () => data.queries.filter((q) => state.filter === "all" || q.category === state.filter);
  const photo = (tag) => data.photos[tag];
  const src = (tag) => assetUrl(`demo/rank_change/${photo(tag).file}`);

  function buildFilters() {
    filterBar.textContent = "";
    for (const [key, label, title] of FILTERS) {
      const n = key === "all" ? data.counts.total : data.counts[key];
      filterBar.append(el("button", {
        class: `wm-demo-tab${state.filter === key ? " is-active" : ""}`, type: "button", role: "tab", title,
        "aria-selected": state.filter === key ? "true" : "false",
        onclick: () => { state.filter = key; state.index = 0; buildFilters(); buildStrip(); render(); },
      }, [el("span", { text: label }), el("small", { text: ` ${n.toLocaleString()} of ${data.counts.total.toLocaleString()} queries` })]));
    }
    const c = data.counts;
    countLine.textContent = `Over all ${c.total.toLocaleString()} queries of the ${data.dataset} test split at k = ${data.k}: cosine retrieval is right at rank 1 for ${c.cosine_top1.toLocaleString()}, default LoMa for ${c.default_top1.toLocaleString()}, LoMa + WildMatch for ${c.finetuned_top1.toLocaleString()}. Fine-tuning rescues ${c.rescued.toLocaleString()} queries and regresses ${c.regressed.toLocaleString()}; ${c.still_wrong.toLocaleString()} remain wrong. The examples below are a random sample of each category, not a selection.`;
  }

  function buildStrip() {
    strip.textContent = "";
    const list = visible();
    list.forEach((q, i) => {
      strip.append(el("button", {
        class: `wm-demo-card${i === state.index ? " is-active" : ""}`, type: "button", role: "listitem",
        "aria-pressed": i === state.index ? "true" : "false",
        title: `${q.identity.replace("_", " ")} · ${FILTERS.find((f) => f[0] === q.category)[1]}`,
        onclick: () => { state.index = i; buildStrip(); render(); },
      }, [
        el("img", { src: src(q.tag), alt: `query photo of ${q.identity}` }),
        el("span", { class: "wm-demo-cap", text: q.identity.replace("lynx_", "lynx ") }),
      ]));
    });
    if (!list.length) strip.append(el("span", { class: "wm-demo-cap", text: "no sampled queries in this category" }));
  }

  function rankText(r) {
    if (r.true_rank == null) return `true individual not in the ${data.k}-candidate shortlist`;
    return r.true_rank === 1 ? "true individual at rank 1" : `true individual first at rank ${r.true_rank}`;
  }

  function render() {
    columns.textContent = "";
    const list = visible();
    if (!list.length) return;
    const q = list[Math.min(state.index, list.length - 1)];
    for (const m of data.methods) {
      const r = q.rankings[m.key];
      const col = el("div", { class: `wm-rc-col${m.key === "finetuned" ? " is-ours" : ""}` });
      col.append(el("div", { class: "wm-rc-head" }, [el("strong", { text: m.label }), el("span", { class: `wm-rc-rank${r.true_rank === 1 ? " is-right" : ""}`, text: rankText(r) })]));
      const row = el("div", { class: "wm-rc-row", role: "list" });
      r.top5.forEach((e, i) => {
        row.append(el("div", { class: `wm-rc-item${e.correct ? " is-correct" : ""}`, role: "listitem",
                             title: `rank ${i + 1}: ${e.identity.replace("_", " ")}, score ${e.score.toFixed(3)}${e.correct ? " (true individual)" : ""}` }, [
          el("img", { src: src(e.tag), alt: `rank ${i + 1}, ${e.identity}${e.correct ? ", the true individual" : ""}` }),
          el("span", { class: "wm-demo-cap", text: `#${i + 1} · ${e.score.toFixed(2)}${e.correct ? " ✓" : ""}` }),
        ]));
      });
      col.append(row);
      columns.append(col);
    }
    caption.textContent = `Query ${q.identity.replace("_", " ")} (${FILTERS.find((f) => f[0] === q.category)[1].toLowerCase()}). Each column is one method's top 5 over the same gallery; blue frames mark gallery photos of the query's own individual. Cosine retrieval scores are cosine similarities of MegaDescriptor-L embeddings; the two LoMa columns re-rank the same ${data.k} MegaDescriptor-L candidates and show matcher scores. ${data.attribution}`;
  }

  buildFilters();
  buildStrip();
  render();
}
