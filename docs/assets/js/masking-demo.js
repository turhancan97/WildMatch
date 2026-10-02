// Background-masking demo: raw render against the SAM 3 masked model input with a
// draggable divider, mask outlines and detection readouts. Data:
// docs/assets/demo/masking/masking_demo.json, written by scripts/export_masking_demo.py.
import { PALETTE, theme, onThemeChange, el, assetUrl, checkbox } from "./wm-common.js";

const MAX_HEIGHT = 440;

export async function mountMaskingDemo(root) {
  let data;
  try {
    const response = await fetch(assetUrl("demo/masking/masking_demo.json"));
    if (!response.ok) throw new Error(`masking_demo.json: HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    root.textContent = "The background-masking demo appears once the SAM 3 masks are exported.";
    root.classList.add("wm-widget--pending");
    return;
  }
  const state = { group: 0, index: 0, split: 0.5, sam3Outline: true, datasetOutline: false, dragging: false };
  root.classList.remove("wm-widget");
  root.textContent = "";

  const groupBar = el("div", { class: "wm-demo-groups", role: "tablist" });
  const strip = el("div", { class: "wm-demo-strip", role: "list", "aria-label": "images" });
  const stage = el("div", { class: "wm-demo-stage wm-mask-stage" });
  const canvas = el("canvas", { class: "wm-match-canvas", role: "img" });
  stage.append(canvas);
  const readout = el("p", { class: "wm-demo-score" });
  const controls = el("div", { class: "wm-controls wm-controls--inline" });
  const caption = el("p", { class: "wm-match-caption" });
  const stripLabel = el("span", { class: "wm-demo-label", text: "Image" });
  root.append(groupBar, el("div", { class: "wm-demo-row" }, [stripLabel, strip]), stage, readout, controls, caption);

  const splitInput = el("input", { type: "range", min: "0", max: "100", step: "1", value: "50", "aria-label": "divider position" });
  splitInput.addEventListener("input", () => { state.split = Number(splitInput.value) / 100; draw(); });
  controls.append(
    el("label", { class: "wm-control" }, [el("span", { text: "Divider: raw | masked input" }), el("span", { class: "wm-range" }, [splitInput])]),
    el("div", { class: "wm-control" }, [el("span", { text: "Outlines" }), el("div", { class: "wm-check-group" }, [
      checkbox("SAM 3 mask", state.sam3Outline, (on) => { state.sam3Outline = on; draw(); }, PALETTE.red),
      checkbox("Reference mask", state.datasetOutline, (on) => { state.datasetOutline = on; draw(); }, PALETTE.blue),
    ])]),
  );

  const cache = new Map();
  function image(file) {
    if (!cache.has(file)) {
      const img = new Image();
      img.addEventListener("load", () => draw(), { once: true });
      img.src = assetUrl(`demo/masking/${file}`);
      cache.set(file, img);
    }
    return cache.get(file);
  }
  const loaded = (img) => img.complete && img.naturalWidth > 0;

  function group() { return data.groups[state.group]; }
  function item() { return group().items[state.index]; }

  function buildGroups() {
    groupBar.textContent = "";
    data.groups.forEach((g, i) => {
      groupBar.append(el("button", {
        class: `wm-demo-tab${i === state.group ? " is-active" : ""}`, type: "button", role: "tab",
        "aria-selected": i === state.group ? "true" : "false",
        onclick: () => { state.group = i; state.index = 0; buildGroups(); buildStrip(); draw(); },
      }, [el("span", { text: g.label }), el("small", { text: ` ${g.summary.items} images · mean IoU ${g.summary.mean_iou_with_reference}` })]));
    });
  }

  function caption_for(it) {
    return group().synthetic ? it.identity.replace("lynx_", "lynx ") : `${it.dataset_label} · ${it.role === "query" ? "query" : "top-1"}`;
  }

  function buildStrip() {
    strip.textContent = "";
    group().items.forEach((it, i) => {
      strip.append(el("button", {
        class: `wm-demo-card${i === state.index ? " is-active" : ""}`, type: "button", role: "listitem",
        "aria-pressed": i === state.index ? "true" : "false",
        title: `${it.dataset_label}, ${it.identity} (${it.role}): SAM 3 score ${it.sam3.score}, IoU ${it.iou_with_dataset_mask}`,
        onclick: () => { state.index = i; buildStrip(); draw(); },
      }, [
        el("img", { src: assetUrl(`demo/masking/${it.image.file}`), alt: `${it.role} image, ${it.dataset_label}, ${it.identity}` }),
        el("span", { class: "wm-demo-cap", text: caption_for(it) }),
      ]));
    });
  }

  // Offscreen helpers: masked input (raw * mask) and mask outlines. The mask PNGs are
  // opaque greyscale, so their brightness is turned into alpha before compositing.
  const work = document.createElement("canvas");
  const workCtx = work.getContext("2d");
  const alphaCache = new Map();

  function alphaMask(mask, w, h) {
    const key = `${mask.src}|${w}x${h}`;
    if (alphaCache.has(key)) return alphaCache.get(key);
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const cx = c.getContext("2d");
    cx.drawImage(mask, 0, 0, w, h);
    const img = cx.getImageData(0, 0, w, h); const px = img.data;
    for (let i = 0; i < px.length; i += 4) { px[i + 3] = px[i]; px[i] = px[i + 1] = px[i + 2] = 255; }
    cx.putImageData(img, 0, 0);
    alphaCache.set(key, c);
    return c;
  }

  function maskedInput(raw, mask, w, h) {
    work.width = w; work.height = h;
    workCtx.globalCompositeOperation = "source-over";
    workCtx.clearRect(0, 0, w, h);
    workCtx.drawImage(raw, 0, 0, w, h);
    workCtx.globalCompositeOperation = "destination-in";
    workCtx.drawImage(alphaMask(mask, w, h), 0, 0);
    workCtx.globalCompositeOperation = "destination-over";
    workCtx.fillStyle = "#000000"; workCtx.fillRect(0, 0, w, h);
    workCtx.globalCompositeOperation = "source-over";
    return work;
  }

  function outlinePath(ctx, mask, w, h) {
    // Edge pixels of the binary mask at display resolution, drawn as small squares.
    work.width = w; work.height = h;
    workCtx.globalCompositeOperation = "source-over";
    workCtx.clearRect(0, 0, w, h);
    workCtx.drawImage(mask, 0, 0, w, h);
    const px = workCtx.getImageData(0, 0, w, h).data;
    const inside = (x, y) => x >= 0 && y >= 0 && x < w && y < h && px[(y * w + x) * 4] > 127;
    ctx.beginPath();
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        if (!inside(x, y)) continue;
        if (!inside(x - 1, y) || !inside(x + 1, y) || !inside(x, y - 1) || !inside(x, y + 1)) ctx.rect(x, y, 1, 1);
      }
    }
  }

  function draw() {
    const it = item();
    const raw = image(it.image.file), sam3 = image(it.sam3_mask), dataset = image(it.dataset_mask);
    if (!(loaded(raw) && loaded(sam3) && loaded(dataset))) return;
    const t = theme();
    const available = Math.max(320, stage.clientWidth || root.clientWidth || 800);
    const scale = Math.min(MAX_HEIGHT / it.image.height, available / it.image.width);
    const w = Math.round(it.image.width * scale), h = Math.round(it.image.height * scale);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const splitX = Math.round(state.split * w);
    // Left: raw render. Right: masked model input.
    ctx.drawImage(raw, 0, 0, w, h);
    const masked = maskedInput(raw, sam3, w, h);
    ctx.drawImage(masked, splitX, 0, w - splitX, h, splitX, 0, w - splitX, h);
    if (state.datasetOutline) { outlinePath(ctx, dataset, w, h); ctx.fillStyle = PALETTE.blue; ctx.fill(); }
    if (state.sam3Outline) { outlinePath(ctx, sam3, w, h); ctx.fillStyle = PALETTE.red; ctx.fill(); }
    // Divider.
    ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.fillRect(splitX - 1, 0, 2, h);
    ctx.beginPath(); ctx.arc(splitX, h / 2, 11, 0, Math.PI * 2); ctx.fillStyle = "rgba(255,255,255,0.92)"; ctx.fill();
    ctx.strokeStyle = PALETTE.greyHead; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = PALETTE.greyHead; ctx.font = "700 11px Roboto, Helvetica, Arial, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("◀ ▶", splitX, h / 2 + 0.5);
    const g = group();
    tag(ctx, 8, 8, g.synthetic ? "Raw render (synthetic)" : `Raw photo · ${it.dataset_label}`);
    tag(ctx, w - 8, 8, "Model input: SAM 3 mask", "right");
    canvas.setAttribute("aria-label", `${it.dataset_label} ${it.identity}: raw image on the left of the divider, SAM 3 masked model input on the right; SAM 3 score ${it.sam3.score}, ${it.sam3.instances} instances, IoU with the reference mask ${it.iou_with_dataset_mask}`);
    readout.textContent = "";
    readout.append(
      el("strong", { text: `Prompt "${it.sam3.prompt}"` }),
      el("span", { text: ` · confidence ${it.sam3.score ?? "–"} · ${it.sam3.instances ?? "–"} instance${it.sam3.instances === 1 ? "" : "s"} merged · foreground ${(it.sam3.foreground_fraction * 100).toFixed(0)} % of the frame · IoU with the reference mask ${it.iou_with_dataset_mask.toFixed(3)} (${it.reference_mask_source})` }),
    );
    caption.textContent = `Drag the divider or use the slider. Red outline: SAM 3 mask; blue outline: the reference mask the pipeline used for this dataset. The model input blacks out every pixel outside the mask; nothing is cropped. ${g.attribution}`;
  }

  function tag(ctx, x, y, text, align = "left") {
    ctx.font = "600 12px Roboto, Helvetica, Arial, sans-serif";
    const w = ctx.measureText(text).width + 12;
    const bx = align === "right" ? x - w : x;
    ctx.fillStyle = "rgba(255,255,255,0.88)"; ctx.fillRect(bx, y, w, 20);
    ctx.fillStyle = PALETTE.greyHead; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.fillText(text, bx + 6, y + 10);
  }

  function pointerSplit(event) {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    state.split = Math.min(1, Math.max(0, x));
    splitInput.value = String(Math.round(state.split * 100));
    draw();
  }
  canvas.addEventListener("pointerdown", (e) => { state.dragging = true; canvas.setPointerCapture(e.pointerId); pointerSplit(e); });
  canvas.addEventListener("pointermove", (e) => { if (state.dragging) pointerSplit(e); });
  canvas.addEventListener("pointerup", (e) => { state.dragging = false; canvas.releasePointerCapture(e.pointerId); });
  canvas.addEventListener("pointercancel", () => { state.dragging = false; });
  canvas.style.touchAction = "none";
  canvas.style.cursor = "ew-resize";

  buildGroups();
  buildStrip();
  draw();
  onThemeChange(draw);
  if (window.ResizeObserver) new ResizeObserver(() => draw()).observe(stage);
}
