// Entry module: mounts each interactive component where its mount point exists.
import { mountExplorer } from "./explorer.js";
import { mountResultsTable } from "./results-table.js";
import { mountTrainingCost } from "./training-cost.js";
import { mountSyntheticDemo } from "./synthetic-demo.js";
import { mountMaskingDemo } from "./masking-demo.js";
import { mountBeforeAfter } from "./before-after-demo.js";
import { mountMinedPairs } from "./mined-pairs-demo.js";
import { mountRankChange } from "./rank-change-demo.js";
import { mountScoreSeparation } from "./score-separation-demo.js";
import { mountBudgetTradeoff } from "./budget-tradeoff.js";
import { mountFrequencyBins } from "./frequency-bins.js";
import { mountChallenge, mountImageQualityTable } from "./data-challenges.js";
import { mountResultsGlance } from "./results-glance.js";
import { mountExplainerVideo } from "./explainer-video.js";
import { mountMatcherAblation } from "./matcher-ablation.js";

const MOUNTS = [
  ["wm-accuracy-explorer", mountExplorer],
  ["wm-results-table", mountResultsTable],
  ["wm-training-cost", mountTrainingCost],
  ["wm-synthetic-demo", mountSyntheticDemo],
  ["wm-masking-demo", mountMaskingDemo],
  ["wm-before-after", mountBeforeAfter],
  ["wm-mined-pairs", mountMinedPairs],
  ["wm-rank-change", mountRankChange],
  ["wm-score-separation", mountScoreSeparation],
  ["wm-budget-tradeoff", mountBudgetTradeoff],
  ["wm-frequency-bins", mountFrequencyBins],
  ["wm-challenge-overexposure", mountChallenge],
  ["wm-challenge-empty_frame", mountChallenge],
  ["wm-challenge-insufficient_detail", mountChallenge],
  ["wm-challenge-corruption", mountChallenge],
  ["wm-challenge-blur", mountChallenge],
  ["wm-challenge-night_infrared", mountChallenge],
  ["wm-challenge-occlusion", mountChallenge],
  ["wm-image-quality-table", mountImageQualityTable],
  ["wm-results-glance", mountResultsGlance],
  ["wm-explainer", mountExplainerVideo],
  ["wm-matcher-ablation", mountMatcherAblation],
];

function mountAll() {
  for (const [id, mount] of MOUNTS) {
    const root = document.getElementById(id);
    if (root && !root.dataset.wmMounted) {
      root.dataset.wmMounted = "1";
      mount(root).catch((error) => {
        root.textContent = `This view failed to initialise (${error.message}).`;
      });
    }
  }
}

// Material exposes document$ for instant navigation; fall back to DOMContentLoaded.
if (window.document$ && typeof window.document$.subscribe === "function") {
  window.document$.subscribe(mountAll);
} else if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", mountAll);
} else {
  mountAll();
}
