// Match viewer: query and top-1 photos with correspondence lines, over
// docs/data/match_examples.json and the photos in docs/assets/match/. The JSON is
// written by `scripts/plot_match_examples.py render --web-export docs/assets/match`.
import { PALETTE, theme, onThemeChange, el, select, mountError, assetUrl } from "./wm-common.js";

const LINE = "#19c2d6"; // cyan, as in the paper figure, outlined in dark for contrast on water

export async function mountMatchViewer(root) {
  let data;
  try {
    const response = await fetch(assetUrl("match/match_examples.json"));
    if (!response.ok) throw new Error(`match_examples.json: HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    root.textContent = "The interactive match viewer appears once the match-example photos are exported.";
    root.classList.add("wm-widget--pending");
    return;
  }
  const examples = data.examples;
  const state = { index: 0, count: 10, mode: "spread" };
  root.classList.remove("wm-widget");
  root.textContent = "";
  const controls = el("div", { class: "wm-controls" });
  const stage = el("div", { class: "wm-match-stage" });
  const canvas = el("canvas", { class: "wm-match-canvas", role: "img" });
  const caption = el("p", { class: "wm-match-caption" });
  stage.append(canvas);
  root.append(controls, stage, caption);

  const countInput = el("input", { type: "range", min: "0", max: "60", step: "1", value: String(state.count), "aria-label": "matches drawn" });
  const countValue = el("span", { class: "wm-range-value", text: String(state.count) });
  countInput.addEventListener("input", () => { state.count = Number(countInput.value); countValue.textContent = countInput.value; draw(); });
  controls.append(
    select("Dataset", examples.map((e, i) => [String(i), e.label]), "0", (v) => { state.index = Number(v); load(); }),
    select("Which matches", [["spread", "Strongest, spatially spread"], ["top", "Strongest"], ["all", "Every match"]], state.mode, (v) => { state.mode = v; draw(); }),
    el("label", { class: "wm-control" }, [el("span", { text: "Matches drawn" }), el("span", { class: "wm-range" }, [countInput, countValue])]),
  );

  const images = { query: new Image(), gallery: new Image() };
  let ready = 0;

  function load() {
    const ex = examples[state.index];
    ready = 0;
    for (const side of ["query", "gallery"]) {
      images[side].onload = () => { if (++ready === 2) draw(); };
      images[side].src = assetUrl(`match/${ex.images[side].file}`);
    }
    countInput.max = String(Math.min(ex.match_count, 120));
    if (state.count > ex.match_count) { state.count = ex.match_count; countInput.value = String(state.count); countValue.textContent = countInput.value; }
  }

  function selection(ex) {
    const order = ex.order_by_confidence;
    if (state.mode === "all") return order;
    if (state.mode === "top" || state.count >= order.length) return order.slice(0, state.count);
    // Greedy spread: take matches in confidence order, skipping any whose endpoints sit
    // closer than a spacing to an already chosen match; halve the spacing until enough.
    const q = ex.points.query, g = ex.points.gallery;
    let spacing = 0.12 * ex.images.query.height;
    while (spacing > 1) {
      const chosen = [];
      for (const i of order) {
        const ok = chosen.every((j) => Math.hypot(q[i][0] - q[j][0], q[i][1] - q[j][1]) >= spacing && Math.hypot(g[i][0] - g[j][0], g[i][1] - g[j][1]) >= spacing);
        if (ok) chosen.push(i);
        if (chosen.length >= state.count) return chosen;
      }
      spacing /= 2;
    }
    return order.slice(0, state.count);
  }

  function draw() {
    if (ready < 2) return;
    const ex = examples[state.index];
    const t = theme();
    const gap = 16;
    const height = 420;
    const scaleQ = height / ex.images.query.height, scaleG = height / ex.images.gallery.height;
    const wQ = Math.round(ex.images.query.width * scaleQ), wG = Math.round(ex.images.gallery.width * scaleG);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = (wQ + gap + wG) * dpr; canvas.height = height * dpr;
    canvas.style.width = `${wQ + gap + wG}px`; canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, wQ + gap + wG, height);
    ctx.drawImage(images.query, 0, 0, wQ, height);
    ctx.drawImage(images.gallery, wQ + gap, 0, wG, height);
    const chosen = selection(ex);
    ctx.lineCap = "round";
    for (const i of chosen) {
      const [qx, qy] = ex.points.query[i], [gx, gy] = ex.points.gallery[i];
      const x1 = qx * scaleQ, y1 = qy * scaleQ, x2 = wQ + gap + gx * scaleG, y2 = gy * scaleG;
      ctx.strokeStyle = "rgba(20,30,40,0.7)"; ctx.lineWidth = 3.2;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.strokeStyle = LINE; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      for (const [x, y] of [[x1, y1], [x2, y2]]) {
        ctx.beginPath(); ctx.arc(x, y, 3.6, 0, Math.PI * 2);
        ctx.fillStyle = LINE; ctx.fill(); ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.2; ctx.stroke();
      }
    }
    tag(ctx, 8, 8, "Query", t); tag(ctx, wQ + gap + 8, 8, "Top-1 match", t);
    tag(ctx, wQ + gap + wG - 8, height - 8, `${ex.match_count} matches`, t, "right", "bottom");
    canvas.setAttribute("aria-label", `${ex.label}: query and top-1 gallery image of the same individual with ${chosen.length} of ${ex.match_count} matches drawn`);
    caption.textContent = `${ex.label}: ${chosen.length} of ${ex.match_count} matches drawn (LoMa + WildMatch, k = 50, pair score ${ex.score.toFixed(3)}). Matching used background-removed inputs; the photos shown are the originals.`;
  }

  function tag(ctx, x, y, text, t, align = "left", baseline = "top") {
    ctx.font = "600 12px Roboto, Helvetica, Arial, sans-serif";
    const w = ctx.measureText(text).width + 12;
    const bx = align === "right" ? x - w : x, by = baseline === "bottom" ? y - 20 : y;
    ctx.fillStyle = "rgba(255,255,255,0.88)"; ctx.fillRect(bx, by, w, 20);
    ctx.fillStyle = PALETTE.greyHead; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.fillText(text, bx + 6, by + 10);
  }

  load();
  onThemeChange(draw);
}
