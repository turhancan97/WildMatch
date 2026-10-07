// Explainer video facade: shows a poster and a play button; on click loads either the
// hosted player (data-youtube="<id>") or a local MP4 (data-src, for the private draft),
// so the home page never loads a player or a video until asked. The transcript under the
// embed comes from docs/data/explainer.json (shot texts and start times).
import { el, assetUrl, loadJson } from "./wm-common.js";

function stamp(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export async function mountExplainerVideo(root) {
  const youtube = (root.dataset.youtube || "").trim();
  const localSrc = (root.dataset.src || "").trim();
  const poster = assetUrl("video/explainer_poster.jpg");
  root.classList.remove("wm-widget");
  root.textContent = "";

  const stage = el("div", { class: "wm-video-stage" });
  const img = el("img", { src: poster, alt: "WildMatch explainer video, 78 seconds", loading: "lazy" });
  const button = el("button", { class: "wm-video-play", type: "button", "aria-label": "Play the explainer video" }, [
    el("span", { class: "wm-video-play-icon", "aria-hidden": "true" }),
    el("span", { text: "Watch the 78-second explainer" }),
  ]);
  stage.append(img, button);
  root.append(stage);

  if (!youtube && !localSrc) {
    button.disabled = true;
    button.lastChild.textContent = "Video coming soon";
  }

  button.addEventListener("click", () => {
    let player;
    if (youtube) {
      player = el("iframe", {
        src: `https://www.youtube-nocookie.com/embed/${encodeURIComponent(youtube)}?autoplay=1&rel=0&cc_load_policy=1`,
        title: "WildMatch explainer", allow: "autoplay; encrypted-media; picture-in-picture", allowfullscreen: "",
        referrerpolicy: "strict-origin-when-cross-origin", loading: "lazy",
      });
    } else {
      player = el("video", { controls: "", autoplay: "", playsinline: "", poster, preload: "metadata" });
      player.append(el("source", { src: localSrc, type: "video/mp4" }));
      const track = root.dataset.track;
      if (track) player.append(el("track", { kind: "captions", srclang: "en", label: "English", src: track, default: "" }));
    }
    stage.textContent = "";
    stage.classList.add("is-playing");
    stage.append(player);
    if (player.play) player.play().catch(() => {});
  });

  try {
    const data = await loadJson("explainer.json");
    const details = el("details", { class: "wm-table-view wm-transcript" }, [el("summary", { text: "Transcript" })]);
    const list = el("ol", { class: "wm-transcript-list" });
    for (const shot of data.shots) {
      list.append(el("li", {}, [el("span", { class: "wm-transcript-time", text: stamp(shot.start) }), el("span", { text: shot.text })]));
    }
    details.append(list, el("p", { class: "wm-table-foot", text: `Narration synthesised with ${data.voice_note}. Photos, correspondences and candidate lists in the video are real outputs from the paper's runs; nothing is drawn by hand.` }));
    root.append(details);
  } catch (error) {
    // the transcript is optional; the video still plays
  }
}
