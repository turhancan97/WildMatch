# WildMatch explainer: narration script and shot list

Status: draft for the user's approval (2026-10-03). Target length 70 to 80 seconds at a
calm 135 words per minute. Narration: Kokoro-82M, stock English voice. Animation: Manim
Community, brand palette (blue #3a7eab for "ours", red #cf4832 for negatives, grey
#d1d3d4 for structure), white background to match the page. No venue, no submission
status, no author names on screen. Numbers spoken: "eight datasets", "about five
GPU-hours"; both are pinned by `tests/test_project_page_numbers.py` once the transcript
is on the page.

## Narration (162 words)

| # | Shot | Seconds | Narration |
|---|---|---|---|
| 1 | Two camera-trap photos of a lynx fade in; keypoints appear on both; a few faint lines connect them. | 0–9 | Telling individual animals apart means comparing the markings in two photos. Image matchers already do this: they find keypoints and connect the ones that correspond. |
| 2 | The lines dim; a label "trained on buildings, streets, landmarks" slides under the matcher; the lynx pair looks ignored. | 9–17 | But these matchers were trained on buildings and streets, not on fur, and wildlife datasets have no keypoint labels to retrain them. |
| 3 | A database of photos tiles in; each gets a small name tag. The tags glow blue. | 17–25 | What every monitoring project does have is identity labels: which photos show the same animal. WildMatch uses nothing else. |
| 4 | One photo becomes the anchor; the pretrained matcher scores it against the others; the five highest same-identity photos gather on the left in blue, the five highest other-identity photos on the right in red. | 25–40 | The pretrained matcher scores each photo against the rest. Its highest-scoring photos of the same animal become positives. Its highest-scoring photos of other animals become hard negatives: the pairs it gets wrong. |
| 5 | A horizontal score axis; a blue dot (positive) and a red dot (negative) sit close; a bracket labelled "margin" appears; the blue dot is pushed right, the red left. Only the "matching module" block highlights; detector and descriptor stay grey. | 40–55 | Training then asks for one thing: the score with a positive must beat the score with a hard negative by a margin. Only the matching module changes; the keypoints and descriptors stay as they were. |
| 6 | Back to the first lynx pair: many more lines appear, all on the coat; a ranked list of candidates reorders so the true match moves to the top. | 55–68 | After adaptation the matcher finds many more correspondences on the animal, and the right individual rises to the top of the candidate list. |
| 7 | Eight small dataset icons with upward arrows; a clock showing five GPU-hours; the WildMatch wordmark. | 68–78 | Across eight wildlife datasets, accuracy improves with about five GPU-hours of training, and every match can be inspected, line by line. |

## Production notes

- Scenes 1 and 6 reuse the Czech Lynx pair of the before/after demo (`docs/assets/demo/before_after/`),
  so the lines drawn are real correspondences from the default and the fine-tuned matcher.
- Scene 4's pools reuse one anchor of the mined-pairs export (`docs/assets/demo/mined_pairs/`),
  so the ten photos and their scores are real mining output.
- Scene 7's eight icons are the datasets in the paper's order; arrows have no numbers.
- Captions: one line per shot, from this table; a plain-text transcript goes under the embed.
- Output: 1920 x 1080, 30 fps, H.264, hosted unlisted; source and script stay in `video/explainer/`.
