# Demo

## Synthetic keypoint matching

Ten synthetic lynxes, two renders each. Choose one of the ten query renders; LoMa +
WildMatch, the matcher fine-tuned on CzechLynx, ranks the ten gallery renders, one per
individual, so exactly one is the same animal. Pick a candidate to see its
correspondences, drag the slider to show more of them, and hover a match to read its
confidence. The matches and scores are real matcher output; the images are synthetic
renders, not the paper's test data, and the scores are not a benchmark result. Synthetic
coats are generated from a shared texture model, so different individuals can look alike
to the matcher and a few queries rank another lynx first by a small margin.

<div id="wm-synthetic-demo" class="wm-widget">Loading the synthetic match demo…</div>

<small>Synthetic lynx renders from the CzechLynx synthetic subset (Picek et al.), Zenodo record 17592004, CC BY 4.0.</small>

## How to read it

- **Score** is the matcher's image score: the summed confidence of the mutual-nearest
  matches above the threshold, divided by the smaller keypoint count, as in the paper.
- **Rank** orders the ten gallery renders by that score for the chosen query. The gallery
  render of the query's own individual carries a red inner frame; hover a thumbnail for
  its details.
- **Lines** are the strongest correspondences; their weight follows confidence, and
  hovering one shows its exact value. Every match is available through the slider.
- **Synthetic**: the images are rendered lynxes, the paper's test data are camera-trap
  photographs. See [Qualitative](qualitative.md) for matches on real photos.

## Background masking with SAM 3

Before feature extraction every image is background-masked, so keypoints land on the
animal rather than on the vegetation or ground that stationary camera traps repeat.
Where a dataset ships masks they are used; otherwise the animal is segmented with
text-prompted SAM 3 and all detected instances are merged into one mask. This demo runs
SAM 3 on the same twenty synthetic renders as the matching demo. Because these renders
also ship with the dataset's own masks, each one reports how well the SAM 3 mask agrees
with it.

<div id="wm-masking-demo" class="wm-widget">Loading the background-masking demo…</div>

- **Divider**: raw render on the left, masked model input on the right. Everything
  outside the mask is set to black; nothing is cropped or recentred.
- **Outlines**: red is the SAM 3 mask, blue the mask shipped with the dataset.
- **Readouts**: the text prompt, SAM 3's detection confidence, how many instances were
  merged, the foreground share of the frame, and the intersection over union with the
  dataset mask.
- **Synthetic**: these are rendered lynxes used because their licence allows display.
  The pipeline's own SAM 3 use is for photographs without provided masks.

<small>Synthetic lynx renders from the CzechLynx synthetic subset (Picek et al.), Zenodo record 17592004, CC BY 4.0.</small>
