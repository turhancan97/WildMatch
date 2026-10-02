# Mined pairs: what weak supervision looks like

What to look for: before fine-tuning, positives and hard negatives often score alike.
That overlap is the only thing the training objective has to remove.
{ .wm-lead }

WildMatch never sees a keypoint label. Its training signal is a set of image pairs mined
once with the *pretrained* matcher on the training split: for each anchor photo, the five
same-individual photos it scores highest become positives and the five other-individual
photos it scores highest become hard negatives. Fine-tuning then pushes the anchor's score
with every positive above its score with every negative. Pick an anchor to see its mined
pools with the pretrained scores; click a photo to see the correspondences behind that
score.

<div id="wm-mined-pairs" class="wm-widget">Loading the mined-pairs browser…</div>

- **Scores** are the pretrained matcher's image scores on background-removed inputs, the
  same quantity the method ranks with. Positives (blue) and hard negatives (red) often
  score alike at this stage: that overlap is what fine-tuning removes.
- **Hard negatives** come from other individuals the pretrained matcher already finds
  similar, frequently from the same camera site.
- **Anchors** shown here were chosen by appearance (daylight colour photos, one per
  individual); the pools and scores are exactly those in the training index.

<small>Photographs from CzechLynx (Picek et al.), time-closed training split.</small>

[Back to the demo overview](index.md){ .md-button }
