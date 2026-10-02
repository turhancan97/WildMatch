# Before and after fine-tuning

What to look for: the same two photos, scored twice. Fine-tuning raises the image score
on every pair, from below 0.42 to above 0.60, and multiplies the correspondences on the animal.
{ .wm-lead }

The most direct view of what WildMatch changes: one photo pair, the query and its correct
top-1 gallery photo of the same individual, matched by the default LoMa matcher and by the
matcher fine-tuned on that dataset. Switch between the two and watch the correspondences
and the image score change. Matches are computed on the background-removed inputs the
paper uses and drawn on the original photos; the pairs are the ones in the paper's
qualitative figure.

<div id="wm-before-after" class="wm-widget" data-pair="0">Loading the before/after demo…</div>

- **Matcher toggle**: the default checkpoint against the checkpoint fine-tuned on the
  shown dataset; each button carries that matcher's score and match count.
- **Pair selector**: one query and top-1 pair for each of the eight datasets in the paper.
- **Slider**: how many of the strongest correspondences are drawn; hover a line for its
  confidence.

This is the only demo that shows the default matcher next to the fine-tuned one. Scores
here can differ from the paper's recorded values by up to 0.02, because the demo extracts
features one image at a time while the benchmark extracts them in batches.

<small>Photographs from CzechLynx (Picek et al.), WildlifeReID-10k and SalamanderID2025
(AnimalCLEF 2025), the latter shown with the dataset team's permission.</small>

[Back to the demo overview](index.md){ .md-button }
