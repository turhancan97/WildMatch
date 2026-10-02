# Qualitative results

## Matches on correct top-1 retrievals

One correct top-1 retrieval per dataset with the fine-tuned LoMa matcher at
\(k = 50\). The query is on the left, its top-1 gallery image of the same individual on
the right. Lines show 10 of the matches, chosen for spatial spread among the most
confident correspondences; the total match count is given in each panel. Matching uses
background-removed inputs, and the matches are drawn on the original photos.

<figure class="wm-figure" markdown>
![LoMa + WildMatch matches on correct top-1 retrievals](assets/figures/match_examples.png)
<figcaption>Panels in alphabetical dataset order: Czech Lynx, Hyena, Leopard, Nyala, Salamander, Sea Star, Turtle, Whale Shark.</figcaption>
</figure>

### Explore the matches

Pick a dataset, choose how many correspondences to draw, and switch between the
strongest matches, a spatially spread selection (as in the figure) and every match.

<div id="wm-match-viewer" class="wm-widget">Loading the match viewer…</div>

How the examples were chosen: candidates are correct top-1 queries from the run's
stored score matrix, excluding pairs from the same encounter or capture day where the
metadata records them, and excluding near-duplicate photos. Candidates were sorted by
MegaDescriptor-L cosine similarity, hardest first, and one pair per dataset was chosen
by eye from a contact sheet. Rank 1 is always read from the run's stored scores, never
recomputed.

## Why this matters

The correspondences link parts of the coat, skin or shell pattern across changes in
viewpoint and illumination: the spots of the hyena and the leopard, the stripes of the
nyala, the yellow patches of the salamander. Because the ranking is built from these
correspondences, every retrieval can be inspected. An expert can check which markings
support a proposed match before accepting it, which a similarity score from a global
embedding alone does not show. In conservation monitoring a wrong identity propagates
into population estimates.

## Challenging images

See [Known data problems](datasets.md#known-data-problems) for the raw-photo examples of
overexposure, insufficient detail, empty frames, corruption and blur that remain in
every method's database and query sets.
