// Mined-pairs browser: an anchor photo, its five mined positives and five hard negatives
// with the pretrained matcher's scores; clicking a partner draws the correspondences
// behind its score. Data: docs/assets/demo/mined_pairs/mined_pairs.json, written by
// scripts/export_mined_pairs_demo.py.
import { PALETTE, theme, onThemeChange, el, assetUrl } from "./wm-common.js";

const LINE = "#19c2d6";
const MAX_HEIGHT = 380;
const GAP = 16;

export async function mountMinedPairs(root) {
  let data;
  try {
    const response = await fetch(assetUrl("demo/mined_pairs/mined_pairs.json"));
    if (!response.ok) throw new Error(`mined_pairs.json: HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    root.textContent = "The mined-pairs browser appears once its data is exported.";
    root.classList.add("wm-widget--pending");
    return;
  }
  const state = { anchor: 0, pool: "positives", index: 0, count: 30, hover: null };
  root.classList.remove("wm-widget");
  root.textContent = "";

  const anchorStrip = el("div", { class: "wm-demo-strip", role: "list", "aria-label": "anchor photos" });
  const pools = el("div", { class: "wm-mp-pools" });
  const stage = el("div", { class: "wm-demo-stage" });
  const canvas = el("canvas", { class: "wm-match-canvas", role: "img" });
  const tooltip = el("div", { class: "wm-demo-tooltip", hidden: "" });
  stage.append(canvas, tooltip);
  const readout = el("p", { class: "wm-demo-score" });
  const caption = el("p", { class: "wm-match-caption" });
  root.append(el("div", { class: "wm-demo-row" }, [el("span", { class: "wm-demo-label", text: "Anchor" }), anchorStrip]), pools, stage, readout, caption);

  const cache = new Map();
  function image(file) {
    if (!cache.has(file)) {
      const img = new Image();
      img.addEventListener("load", () => draw(), { once: true });
      img.src = assetUrl(`demo/mined_pairs/${file}`);
      cache.set(file, img);
    }
    return cache.get(file);
  }
  const loaded = (img) => img.complete && img.naturalWidth > 0;
  const anchor = () => data.anchors[state.anchor];
  const partner = () => anchor()[state.pool][state.index];

  function buildAnchorStrip() {
    anchorStrip.textContent = "";
    data.anchors.forEach((a, i) => {
      anchorStrip.append(el("button", {
        class: `wm-demo-card${i === state.anchor ? " is-active" : ""}`, type: "button", role: "listitem",
        "aria-pressed": i === state.anchor ? "true" : "false",
        title: `${a.identity.replace("_", " ")} at ${a.site}; anchor kept with best pair score ${a.selection_score}`,
        onclick: () => { state.anchor = i; state.pool = "positives"; state.index = 0; state.hover = null; buildAnchorStrip(); buildPools(); draw(); },
      }, [
        el("img", { src: assetUrl(`demo/mined_pairs/${a.anchor.image.file}`), alt: `anchor photo of ${a.identity}` }),
        el("span", { class: "wm-demo-cap", text: a.identity.replace("lynx_", "lynx ") }),
      ]));
    });
  }

  function buildPools() {
    pools.textContent = "";
    const a = anchor();
    const maxScore = Math.max(...a.positives.map((p) => p.mined_score), ...a.negatives.map((p) => p.mined_score), 1e-6);
    for (const [pool, label, cls] of [["positives", "Mined positives · same individual, highest pretrained scores", "is-pos"],
                                      ["negatives", "Hard negatives · other individuals, highest pretrained scores", "is-neg"]]) {
      const row = el("div", { class: "wm-mp-row", role: "list" });
      a[pool].forEach((p, i) => {
        const active = state.pool === pool && state.index === i;
        row.append(el("button", {
          class: `wm-mp-card ${cls}${active ? " is-active" : ""}`, type: "button", role: "listitem", "aria-pressed": active ? "true" : "false",
          title: `${p.identity.replace("_", " ")}, ${p.collection}: mined score ${p.mined_score}${p.match_count != null ? `, ${p.match_count} matches` : ""}`,
          onclick: () => { state.pool = pool; state.index = i; state.hover = null; buildPools(); draw(); },
        }, [
          el("img", { src: assetUrl(`demo/mined_pairs/${p.image.file}`), alt: `${pool === "positives" ? "positive" : "hard negative"} ${p.identity}` }),
          el("span", { class: "wm-mp-bar" }, [el("span", { class: "wm-mp-fill", style: `width:${Math.max(3, 100 * p.mined_score / maxScore)}%` })]),
          el("span", { class: "wm-demo-cap", text: `${p.mined_score.toFixed(3)} · ${p.identity.replace("lynx_", "lynx ")}` }),
        ]));
      });
      pools.append(el("div", { class: "wm-mp-pool" }, [el("span", { class: `wm-demo-label ${cls}`, text: label }), row]));
    }
  }

  let geom = null;

  function draw() {
    const a = anchor(), p = partner();
    const qImg = image(a.anchor.image.file), gImg = image(p.image.file);
    if (!(loaded(qImg) && loaded(gImg))) return;
    const t = theme();
    const available = Math.max(320, stage.clientWidth || root.clientWidth || 800);
    const aspectSum = a.anchor.image.width / a.anchor.image.height + p.image.width / p.image.height;
    const height = Math.min(MAX_HEIGHT, Math.floor((available - GAP) / aspectSum));
    const sQ = height / a.anchor.image.height, sG = height / p.image.height;
    const wQ = Math.round(a.anchor.image.width * sQ), wG = Math.round(p.image.width * sG);
    const width = wQ + GAP + wG;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr;
    canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(qImg, 0, 0, wQ, height);
    ctx.drawImage(gImg, wQ + GAP, 0, wG, height);
    const segments = [];
    let drawn = 0;
    if (p.points) {
      const chosen = p.order_by_confidence.slice(0, Math.min(state.count, p.match_count));
      const maxConf = p.confidence.length ? Math.max(...p.confidence) : 1;
      ctx.lineCap = "round";
      for (const i of chosen) {
        const [qx, qy] = p.points.anchor[i], [gx, gy] = p.points.partner[i];
        const x1 = qx * sQ, y1 = qy * sQ, x2 = wQ + GAP + gx * sG, y2 = gy * sG;
        const conf = p.confidence[i];
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
      drawn = chosen.length;
    }
    const kind = state.pool === "positives" ? "Mined positive (same individual)" : "Hard negative (other individual)";
    tag(ctx, 8, 8, `Anchor · ${a.identity.replace("_", " ")}`); tag(ctx, wQ + GAP + 8, 8, `${kind} · ${p.identity.replace("_", " ")}`);
    geom = { segments };
    canvas.setAttribute("aria-label", `${kind} for anchor ${a.identity}: pretrained matcher score ${p.mined_score}${p.match_count != null ? `, ${p.match_count} matches, ${drawn} drawn` : ""}`);
    readout.textContent = "";
    readout.append(
      el("strong", { text: `Pretrained matcher score ${p.mined_score.toFixed(3)}` }),
      el("span", { text: p.match_count != null ? ` · ${p.match_count} mutual matches (recomputed ${p.recomputed_score.toFixed(3)}) · ${drawn} drawn, strongest first` : "" }),
    );
    caption.textContent = `Anchor ${a.identity.replace("_", " ")} photographed at ${a.site}. Mining kept the five same-individual photos and the five other-individual photos the pretrained matcher scored highest; fine-tuning then pushes the anchor's score with each positive above its score with each negative. Hover a match to read its confidence. ${data.attribution}`;
  }

  function tag(ctx, x, y, text) {
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

  buildAnchorStrip();
  buildPools();
  draw();
  onThemeChange(draw);
  if (window.ResizeObserver) new ResizeObserver(() => draw()).observe(stage);
}
