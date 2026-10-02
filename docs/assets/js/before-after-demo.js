// Before vs after fine-tuning on the same photo pair: toggle between the default LoMa
// matcher and LoMa + WildMatch. Data: docs/assets/demo/before_after/before_after.json,
// written by scripts/export_before_after_demo.py (matches computed on background-removed
// inputs, drawn on the original photos).
import { PALETTE, theme, onThemeChange, el, assetUrl, select } from "./wm-common.js";

const LINE = "#19c2d6";
const MAX_HEIGHT = 420;
const GAP = 16;
const MATCHERS = [["default", "Default LoMa"], ["finetuned", "LoMa + WildMatch"]];

export async function mountBeforeAfter(root) {
  let data;
  try {
    const response = await fetch(assetUrl("demo/before_after/before_after.json"));
    if (!response.ok) throw new Error(`before_after.json: HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    root.textContent = "The before/after demo appears once its matches are exported.";
    root.classList.add("wm-widget--pending");
    return;
  }
  const defaultPair = Number(root.dataset.pair || 0);
  const state = { pair: Math.min(defaultPair, data.pairs.length - 1), matcher: "finetuned", count: 30, hover: null };
  root.classList.remove("wm-widget");
  root.textContent = "";

  const toggle = el("div", { class: "wm-ba-toggle", role: "tablist", "aria-label": "matcher" });
  const stage = el("div", { class: "wm-demo-stage" });
  const canvas = el("canvas", { class: "wm-match-canvas", role: "img" });
  const tooltip = el("div", { class: "wm-demo-tooltip", hidden: "" });
  stage.append(canvas, tooltip);
  const scores = el("div", { class: "wm-ba-scores" });
  const controls = el("div", { class: "wm-controls wm-controls--inline" });
  const caption = el("p", { class: "wm-match-caption" });
  root.append(toggle, stage, scores, controls, caption);

  const countInput = el("input", { type: "range", min: "0", max: "60", step: "1", value: String(state.count), "aria-label": "matches drawn" });
  const countValue = el("span", { class: "wm-range-value", text: String(state.count) });
  countInput.addEventListener("input", () => { state.count = Number(countInput.value); countValue.textContent = countInput.value; draw(); });
  controls.append(
    select("Photo pair", data.pairs.map((p, i) => [String(i), p.label]), String(state.pair), (v) => { state.pair = Number(v); state.hover = null; load(); }),
    el("label", { class: "wm-control" }, [el("span", { text: "Strongest matches drawn" }), el("span", { class: "wm-range" }, [countInput, countValue])]),
  );

  const cache = new Map();
  function image(file) {
    if (!cache.has(file)) {
      const img = new Image();
      img.addEventListener("load", () => draw(), { once: true });
      img.src = assetUrl(`demo/before_after/${file}`);
      cache.set(file, img);
    }
    return cache.get(file);
  }
  const loaded = (img) => img.complete && img.naturalWidth > 0;
  const pair = () => data.pairs[state.pair];
  const result = () => pair().results[state.matcher];

  function buildToggle() {
    toggle.textContent = "";
    for (const [key, label] of MATCHERS) {
      const r = pair().results[key];
      toggle.append(el("button", {
        class: `wm-ba-button${state.matcher === key ? " is-active" : ""}${key === "finetuned" ? " is-ours" : ""}`,
        type: "button", role: "tab", "aria-selected": state.matcher === key ? "true" : "false",
        onclick: () => { state.matcher = key; state.hover = null; buildToggle(); draw(); },
      }, [el("span", { class: "wm-ba-name", text: label }), el("span", { class: "wm-ba-stat", text: `score ${r.score.toFixed(3)} · ${r.match_count} matches` })]));
    }
  }

  function load() {
    const maxCount = Math.max(pair().results.default.match_count, pair().results.finetuned.match_count);
    countInput.max = String(Math.min(maxCount, 150));
    if (state.count > maxCount) state.count = maxCount;
    countInput.value = String(state.count); countValue.textContent = countInput.value;
    buildToggle();
    draw();
  }

  let geom = null;

  function draw() {
    const p = pair(), r = result();
    const qImg = image(p.query.image.file), gImg = image(p.gallery.image.file);
    if (!(loaded(qImg) && loaded(gImg))) return;
    const t = theme();
    const available = Math.max(320, stage.clientWidth || root.clientWidth || 800);
    const aspectSum = p.query.image.width / p.query.image.height + p.gallery.image.width / p.gallery.image.height;
    const height = Math.min(MAX_HEIGHT, Math.floor((available - GAP) / aspectSum));
    const sQ = height / p.query.image.height, sG = height / p.gallery.image.height;
    const wQ = Math.round(p.query.image.width * sQ), wG = Math.round(p.gallery.image.width * sG);
    const width = wQ + GAP + wG;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr;
    canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(qImg, 0, 0, wQ, height);
    ctx.drawImage(gImg, wQ + GAP, 0, wG, height);
    const chosen = r.order_by_confidence.slice(0, Math.min(state.count, r.match_count));
    const maxConf = r.confidence.length ? Math.max(...r.confidence) : 1;
    const segments = [];
    ctx.lineCap = "round";
    for (const i of chosen) {
      const [qx, qy] = r.points.query[i], [gx, gy] = r.points.gallery[i];
      const x1 = qx * sQ, y1 = qy * sQ, x2 = wQ + GAP + gx * sG, y2 = gy * sG;
      const conf = r.confidence[i];
      const emphasis = state.hover === i ? 1.0 : 0.45 + 0.55 * (conf / maxConf);
      segments.push({ i, x1, y1, x2, y2, conf });
      ctx.globalAlpha = state.hover == null || state.hover === i ? 1 : 0.35;
      ctx.strokeStyle = "rgba(20,30,40,0.7)"; ctx.lineWidth = 3.4 * emphasis + 0.6;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.strokeStyle = state.hover === i ? PALETTE.red : LINE; ctx.lineWidth = 1.8 * emphasis + 0.4;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      for (const [x, y] of [[x1, y1], [x2, y2]]) {
        ctx.beginPath(); ctx.arc(x, y, state.hover === i ? 5 : 3.6, 0, Math.PI * 2);
        ctx.fillStyle = state.hover === i ? PALETTE.red : LINE; ctx.fill();
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.2; ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    const name = MATCHERS.find(([k]) => k === state.matcher)[1];
    tag(ctx, 8, 8, "Query"); tag(ctx, wQ + GAP + 8, 8, "Top-1 match (same individual)");
    tag(ctx, width - 8, height - 8, `${name} · ${r.match_count} matches · score ${r.score.toFixed(3)}`, "right", "bottom");
    geom = { segments };
    canvas.setAttribute("aria-label", `${p.label}: query and top-1 gallery photo of the same individual; ${name} finds ${r.match_count} matches with score ${r.score.toFixed(3)}; ${chosen.length} drawn`);
    scores.textContent = "";
    for (const [key, label] of MATCHERS) {
      const rr = p.results[key];
      scores.append(el("div", { class: `wm-ba-score${key === state.matcher ? " is-active" : ""}` }, [
        el("span", { class: "wm-ba-score-name", text: label }),
        el("strong", { text: rr.score.toFixed(3) }),
        el("span", { text: ` score · ${rr.match_count} matches` }),
      ]));
    }
    const d = p.results.default, f = p.results.finetuned;
    caption.textContent = `${p.label}: fine-tuning changes the score from ${d.score.toFixed(3)} to ${f.score.toFixed(3)} and the match count from ${d.match_count} to ${f.match_count} on the same pair. Matches are computed on background-removed inputs and drawn on the original photos; ${chosen.length} of ${r.match_count} shown, strongest first, line weight follows confidence. ${data.attribution}`;
  }

  function tag(ctx, x, y, text, align = "left", baseline = "top") {
    ctx.font = "600 12px Roboto, Helvetica, Arial, sans-serif";
    const w = ctx.measureText(text).width + 12;
    const bx = align === "right" ? x - w : x, by = baseline === "bottom" ? y - 20 : y;
    ctx.fillStyle = "rgba(255,255,255,0.88)"; ctx.fillRect(bx, by, w, 20);
    ctx.fillStyle = PALETTE.greyHead; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.fillText(text, bx + 6, by + 10);
  }

  function nearest(px, py) {
    if (!geom) return null;
    let best = null, bestD = 14;
    for (const s of geom.segments) {
      for (const [x, y] of [[s.x1, s.y1], [s.x2, s.y2]]) {
        const dd = Math.hypot(px - x, py - y);
        if (dd < bestD) { bestD = dd; best = s; }
      }
    }
    return best;
  }
  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left, py = event.clientY - rect.top;
    const hit = nearest(px, py);
    const next = hit ? hit.i : null;
    if (next !== state.hover) { state.hover = next; draw(); }
    if (hit) { tooltip.hidden = false; tooltip.textContent = `match confidence ${hit.conf.toFixed(3)}`; tooltip.style.left = `${px + 12}px`; tooltip.style.top = `${py - 28}px`; }
    else tooltip.hidden = true;
  });
  canvas.addEventListener("mouseleave", () => { tooltip.hidden = true; if (state.hover != null) { state.hover = null; draw(); } });

  load();
  onThemeChange(draw);
  if (window.ResizeObserver) new ResizeObserver(() => draw()).observe(stage);
}
