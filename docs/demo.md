# Demo

## Before and after fine-tuning

The most direct view of what WildMatch changes: one photo pair, the query and its correct
top-1 gallery photo of the same individual, matched by the default LoMa matcher and by the
matcher fine-tuned on that dataset. Switch between the two and watch the correspondences
and the image score change. Matches are computed on the background-removed inputs the
paper uses and drawn on the original photos; the pairs are the ones in the paper's
qualitative figure.

<div id="wm-before-after" class="wm-widget" data-pair="0">Loading the before/after demo…</div>

## Mined pairs: what weak supervision looks like

WildMatch never sees a keypoint label. Its training signal is a set of image pairs mined
once with the *pretrained* matcher on the training split: for each anchor photo, the five
same-individual photos it scores highest become positives and the five other-individual
photos it scores highest become hard negatives. Fine-tuning then pushes the anchor's score
with every positive above its score with every negative. Pick an anchor to see its mined
pools with the pretrained scores; click a photo to see the correspondences behind that
score.

<div id="wm-mined-pairs" class="wm-widget">Loading the mined-pairs browser…</div>

- **Scores** are the pretrained matcher's image scores on background-removed inputs, the
  same quantity the method ranks with. Positives and hard negatives often score alike at
  this stage: that overlap is what fine-tuning removes.
- **Hard negatives** come from other individuals the pretrained matcher already finds
  similar, frequently from the same camera site.
- **Anchors** shown here were chosen by appearance (daylight colour photos, one per
  individual); the pools and scores are exactly those in the training index.

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
SAM 3 with the prompt "animal" on two groups of images:

- **Synthetic renders**: the twenty renders of the matching demo. The synthetic subset
  ships segmentation masks, so each SAM 3 mask is compared with them.
- **Camera-trap and field photographs**: the query and top-1 photo of every dataset in
  the paper, the same pairs as the match figure. Only CzechLynx ships segmentation masks,
  so the comparison appears for its two photos; for the other datasets the masks used in
  the paper were computed by us and the demo shows the SAM 3 mask alone.

<div id="wm-masking-demo" class="wm-widget">Loading the background-masking demo…</div>

- **Divider**: raw image on the left, masked model input on the right. Everything
  outside the mask is set to black; nothing is cropped or recentred.
- **Outlines**: red is the SAM 3 mask; blue, where available, the segmentation mask
  shipped with the dataset.
- **Readouts**: the text prompt actually used, SAM 3's detection confidence, how many
  instances were merged, the foreground share of the frame, and, where a dataset mask
  exists, the intersection over union with it.
- **Where the masks disagree**: on snow renders the dataset mask includes the animal's
  cast shadow and SAM 3 does not. Sea stars are not found by the prompt "animal" at all;
  the pipeline's fallback prompt "sea star" then segments them, and the readout shows it.

<small>Synthetic lynx renders from the CzechLynx synthetic subset (Picek et al.), Zenodo record 17592004, CC BY 4.0. Photographs from CzechLynx, WildlifeReID-10k and SalamanderID2025 (AnimalCLEF 2025), the latter shown with the dataset team's permission.</small>
