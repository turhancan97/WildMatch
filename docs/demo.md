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
- **Rank** orders the ten gallery renders by that score for the chosen query.
- **Lines** are the strongest correspondences; their weight follows confidence, and
  hovering one shows its exact value. Every match is available through the slider.
- **Synthetic**: the images are rendered lynxes, the paper's test data are camera-trap
  photographs. See [Qualitative](qualitative.md) for matches on real photos.
