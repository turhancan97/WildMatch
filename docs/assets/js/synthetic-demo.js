// Synthetic keypoint-match demo: ten CzechLynx synthetic individuals, two renders each.
// Visitors pick one of the ten query renders; its ten gallery candidates (one per
// individual) are shown ranked by LoMa + WildMatch, with every correspondence and its
// confidence. Data: docs/assets/demo/synthetic/synthetic_demo.json, written by
// scripts/export_synthetic_demo.py from real matcher output.
import { PALETTE, theme, onThemeChange, el, assetUrl } from "./wm-common.js";

const LINE = "#19c2d6";
const PHOTO_HEIGHT = 380;
const GAP = 18;

export async function mountSyntheticDemo(root) {
  let data;
  try {
    const response = await fetch(assetUrl("demo/synthetic/synthetic_demo.json"));
    if (!response.ok) throw new Error(`synthetic_demo.json: HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    root.textContent = "The synthetic match demo appears once its matches are exported.";
    root.classList.add("wm-widget--pending");
    return;
  }
  const state = { query: 0, index: 0, count: 10, hover: null };
  root.classList.remove("wm-widget");
  root.textContent = "";

  const queryStrip = el("div", { class: "wm-demo-strip wm-demo-strip--queries", role: "list", "aria-label": "query renders" });
  const strip = el("div", { class: "wm-demo-strip", role: "list", "aria-label": "ranked gallery renders" });
  const stage = el("div", { class: "wm-demo-stage" });
  const canvas = el("canvas", { class: "wm-match-canvas", role: "img" });
  const tooltip = el("div", { class: "wm-demo-tooltip", hidden: "" });
  stage.append(canvas, tooltip);
  const scoreLine = el("p", { class: "wm-demo-score" });
  const controls = el("div", { class: "wm-controls wm-controls--inline" });
  const caption = el("p", { class: "wm-match-caption" });
  root.append(
    el("p", { class: "wm-demo-label", text: "1. Choose a query render" }), queryStrip,
    el("p", { class: "wm-demo-label", text: "2. Gallery, ranked by matching score (one render per individual)" }), strip,
    stage, scoreLine, controls, caption,
  );

  const countInput = el("input", { type: "range", min: "0", max: "10", step: "1", value: "10", "aria-label": "matches drawn" });
  const countValue = el("span", { class: "wm-range-value", text: "10" });
  countInput.addEventListener("input", () => { state.count = Number(countInput.value); countValue.textContent = countInput.value; draw(); });
  controls.append(el("label", { class: "wm-control" }, [el("span", { text: "Strongest matches drawn" }), el("span", { class: "wm-range" }, [countInput, countValue])]));

  // One Image per file; a load triggers a redraw once. Never call back synchronously
  // for an already loaded image, or draw() would recurse into itself.
  const imageCache = new Map();
  function image(file) {
    if (!imageCache.has(file)) {
      const img = new Image();
      img.addEventListener("load", () => draw(), { once: true });
      img.src = assetUrl(`demo/synthetic/${file}`);
      imageCache.set(file, img);
    }
    return imageCache.get(file);
  }
  const loaded = (img) => img.complete && img.naturalWidth > 0;

  function query() { return data.queries[state.query]; }
  function candidate() { return query().candidates[state.index]; }

  function buildQueryStrip() {
    queryStrip.textContent = "";
    data.queries.forEach((q, i) => {
      queryStrip.append(el("button", {
        class: `wm-demo-card${i === state.query ? " is-active" : ""}`, type: "button", role: "listitem",
        "aria-pressed": i === state.query ? "true" : "false",
        onclick: () => { state.query = i; state.index = 0; state.hover = null; buildQueryStrip(); buildStrip(); load(); },
      }, [
        el("img", { src: assetUrl(`demo/synthetic/${q.image.file}`), alt: `Query render of ${q.identity}` }),
        el("span", { class: "wm-demo-rank", text: q.identity.replace("_", " ") }),
        el("span", { class: `wm-demo-verdict ${q.correct_rank === 1 ? "is-same" : ""}`, text: q.correct_rank === 1 ? "correct at rank 1" : `correct at rank ${q.correct_rank}` }),
      ]));
    });
  }

  function buildStrip() {
    strip.textContent = "";
    query().candidates.forEach((c, i) => {
      strip.append(el("button", {
        class: `wm-demo-card${i === state.index ? " is-active" : ""}`, type: "button", role: "listitem",
        "aria-pressed": i === state.index ? "true" : "false",
        onclick: () => { state.index = i; state.hover = null; buildStrip(); load(); },
      }, [
        el("img", { src: assetUrl(`demo/synthetic/${c.image.file}`), alt: `Gallery render ${c.rank}, ${c.identity}` }),
        el("span", { class: "wm-demo-rank", text: `#${c.rank}` }),
        el("span", { class: "wm-demo-cardscore", text: `score ${c.score.toFixed(3)}` }),
        el("span", { class: `wm-demo-verdict ${c.same_individual ? "is-same" : ""}`, text: c.same_individual ? "same individual" : "other individual" }),
      ]));
    });
  }

  function load() {
    const c = candidate();
    countInput.max = String(c.match_count);
    if (state.count > c.match_count) state.count = c.match_count;
    if (state.count === 0 && c.match_count > 0) state.count = Math.min(10, c.match_count);
    countInput.value = String(state.count); countValue.textContent = countInput.value;
    draw();
  }

  let geom = null;

  function draw() {
    const q = query(), c = candidate();
    const qImg = image(q.image.file), gImg = image(c.image.file);
    if (!(loaded(qImg) && loaded(gImg))) return;
    const t = theme();
    const sQ = PHOTO_HEIGHT / q.image.height, sG = PHOTO_HEIGHT / c.image.height;
    const wQ = Math.round(q.image.width * sQ), wG = Math.round(c.image.width * sG);
    const width = wQ + GAP + wG;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = PHOTO_HEIGHT * dpr;
    canvas.style.width = `${width}px`; canvas.style.height = `${PHOTO_HEIGHT}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, PHOTO_HEIGHT);
    ctx.drawImage(qImg, 0, 0, wQ, PHOTO_HEIGHT);
    ctx.drawImage(gImg, wQ + GAP, 0, wG, PHOTO_HEIGHT);
    const chosen = c.order_by_confidence.slice(0, state.count);
    const segments = [];
    const maxConf = c.confidence.length ? Math.max(...c.confidence) : 1;
    ctx.lineCap = "round";
    for (const i of chosen) {
      const [qx, qy] = c.points.query[i], [gx, gy] = c.points.gallery[i];
      const x1 = qx * sQ, y1 = qy * sQ, x2 = wQ + GAP + gx * sG, y2 = gy * sG;
      const conf = c.confidence[i];
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
    tag(ctx, 8, 8, `Query ${q.identity.replace("_", " ")} (synthetic)`, t);
    tag(ctx, wQ + GAP + 8, 8, `Rank ${c.rank}: ${c.identity.replace("_", " ")} (synthetic)`, t);
    geom = { segments };
    canvas.setAttribute("aria-label", `Synthetic query render of ${q.identity} against gallery render of ${c.identity}, ranked ${c.rank} with score ${c.score.toFixed(3)}, ${chosen.length} of ${c.match_count} matches drawn`);
    scoreLine.textContent = "";
    scoreLine.append(
      el("strong", { text: `Score ${c.score.toFixed(3)}` }),
      el("span", { text: ` · ${c.match_count} matches · rank ${c.rank} of ${q.candidates.length} · ` }),
      el("span", { class: `wm-demo-verdict ${c.same_individual ? "is-same" : ""}`, text: c.same_individual ? "same individual as the query" : "a different individual" }),
    );
    caption.textContent = `${chosen.length} of ${c.match_count} matches drawn, strongest first; line weight follows confidence. Hover a match to read its confidence. ${data.attribution}`;
  }

  function tag(ctx, x, y, text, t) {
    ctx.font = "600 12px Roboto, Helvetica, Arial, sans-serif";
    const w = ctx.measureText(text).width + 12;
    ctx.fillStyle = "rgba(255,255,255,0.88)"; ctx.fillRect(x, y, w, 20);
    ctx.fillStyle = PALETTE.greyHead; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.fillText(text, x + 6, y + 10);
  }

  function nearest(px, py) {
    if (!geom) return null;
    let best = null, bestD = 14;
    for (const s of geom.segments) {
      for (const [x, y] of [[s.x1, s.y1], [s.x2, s.y2]]) {
        const d = Math.hypot(px - x, py - y);
        if (d < bestD) { bestD = d; best = s; }
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
    if (hit) {
      tooltip.hidden = false;
      tooltip.textContent = `match confidence ${hit.conf.toFixed(3)}`;
      tooltip.style.left = `${px + 12}px`; tooltip.style.top = `${py - 28}px`;
    } else {
      tooltip.hidden = true;
    }
  });
  canvas.addEventListener("mouseleave", () => { tooltip.hidden = true; if (state.hover != null) { state.hover = null; draw(); } });

  buildQueryStrip();
  buildStrip();
  load();
  onThemeChange(draw);
}
