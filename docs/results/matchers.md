# Other matchers

!!! note "Extension, not part of the paper"
    The paper compares LoMa and RDD-LightGlue. This page adds two more keypoint matchers,
    ALIKED-LightGlue and SuperPoint-LightGlue, so you can see where the paper's choice of matcher
    stands. The ALIKED-LightGlue and SuperPoint-LightGlue results do not appear in the paper.

All four matchers run with their **default (pretrained) weights**, with no fine-tuning, on the
eight paper datasets and the paper's input images. Each one scores the same MegaDescriptor-L
candidate list as in the paper, at every candidate budget \(k\) of the paper's grid. The LoMa and
RDD-LightGlue values are the paper's default-matcher results; the ALIKED-LightGlue and
SuperPoint-LightGlue runs were added afterwards on RTX 4090 GPUs.

<div id="wm-matcher-ablation" class="wm-widget">Loading the matcher comparison…</div>

## At the paper's budget, \(k = 250\)

Top-1 / top-5 / balanced top-1 in %, default weights; the best top-1 per dataset is bold.

<div class="wm-numeric" markdown>

| Dataset | LoMa | RDD-LightGlue | SuperPoint-LightGlue | ALIKED-LightGlue |
|---|---:|---:|---:|---:|
| CzechLynx | 46.0 / 55.2 / 31.3 | **47.6 / 56.0 / 33.4** | 35.1 / 46.8 / 24.7 | 30.4 / 43.8 / 23.4 |
| Hyena | 88.4 / 91.3 / 82.7 | **90.8 / 91.9 / 85.7** | 74.8 / 84.4 / 65.1 | 72.2 / 82.9 / 67.3 |
| Leopard | 81.7 / 84.2 / 54.6 | **82.3 / 84.2 / 55.0** | 75.8 / 79.8 / 45.3 | 81.1 / 84.4 / 54.0 |
| Nyala | **62.9 / 72.7 / 48.5** | 61.2 / 70.1 / 48.5 | 48.4 / 62.4 / 37.1 | 50.0 / 62.4 / 37.8 |
| Salamander | **53.7 / 55.7 / 56.2** | 42.3 / 43.9 / 43.7 | 43.5 / 44.3 / 44.9 | 36.2 / 40.7 / 37.4 |
| Sea star | **96.3 / 97.2 / 97.1** | 93.7 / 94.6 / 94.5 | 90.9 / 93.0 / 92.4 | 93.0 / 94.9 / 94.0 |
| Whale shark | **70.0 / 74.7 / 59.5** | 63.6 / 68.5 / 52.0 | 68.8 / 73.9 / 58.5 | 56.4 / 64.4 / 47.5 |
| Turtle | **83.5 / 89.9 / 81.4** | 50.5 / 56.5 / 38.7 | 45.1 / 51.3 / 32.5 | 39.5 / 52.3 / 34.7 |
| **Mean** | 72.8 / 77.6 / 63.9 | 66.5 / 70.7 / 56.4 | 60.3 / 67.0 / 50.1 | 57.3 / 65.7 / 49.5 |

</div>

## What the comparison shows

- **LoMa is the strongest matcher on average** (mean top-1 72.8 % at \(k = 250\)) and the best
  on five of the eight datasets. RDD-LightGlue is slightly ahead on CzechLynx, Hyena and Leopard
  (by 0.6 to 2.4 points of top-1).
- **ALIKED-LightGlue and SuperPoint-LightGlue trail both** on average (57.3 % and 60.3 %). They
  come close only in single cases: ALIKED on Leopard and Sea star, SuperPoint on Whale shark.
- **A larger candidate list does not always help.** For LoMa and RDD-LightGlue, accuracy mostly
  grows with \(k\); for ALIKED and SuperPoint it stops growing early and falls on several
  datasets (Hyena, Turtle), because more gallery images that do not show the individual can then
  outscore the right one.
- **Cost is similar.** Feature matching takes about 2 ms per candidate pair for both added
  matchers (median 1.9 ms SuperPoint, 2.4 ms ALIKED), in the same range as LoMa and RDD-LightGlue.

Fine-tuning, the subject of the paper, exists only for LoMa and RDD-LightGlue; whether it would
help the other two matchers is untested.

## Reproduce

```bash
wildmatch sweep matcher_ablation --submit        # k = 250
wildmatch sweep matcher_ablation_grid --submit   # k = 10, 50, 100, 500, 1000
```

Both sweeps write to `experiments/matcher-ablation/`, and
`python paper/page/export_matcher_ablation.py` rebuilds this page's data.
