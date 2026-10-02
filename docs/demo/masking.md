# Background masking with SAM 3

What to look for: everything outside the mask is black in the model input, so a
correspondence can only land on the animal.
{ .wm-lead }

Before feature extraction every image is background-masked, so keypoints land on the
animal rather than on the vegetation or ground that stationary camera traps repeat.
Where a dataset ships masks they are used; otherwise the animal is segmented with
text-prompted SAM 3 and all detected instances are merged into one mask. This demo runs
SAM 3 with the prompt "animal" on two groups of images:

- **Synthetic renders**: the twenty renders of the [synthetic matching demo](synthetic.md).
  The synthetic subset ships segmentation masks, so each SAM 3 mask is compared with them.
- **Camera-trap and field photographs**: the query and top-1 photo of every dataset in
  the paper, the same pairs as the [before/after demo](before-after.md). Only CzechLynx
  ships segmentation masks, so the comparison appears for its two photos; for the other
  datasets the masks used in the paper were computed by us and the demo shows the SAM 3
  mask alone.

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

<small>Synthetic lynx renders from the CzechLynx synthetic subset (Picek et al.), Zenodo
record 17592004, CC BY 4.0. Photographs from CzechLynx, WildlifeReID-10k and
SalamanderID2025 (AnimalCLEF 2025), the latter shown with the dataset team's permission.</small>

[Back to the demo overview](index.md){ .md-button }
