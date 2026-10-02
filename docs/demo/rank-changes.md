# Rank changes: where fine-tuning helps and where it does not

What to look for: the counts are over the whole CzechLynx test split, and the examples
are a random sample, so failures are shown as often as they occur.
{ .wm-lead }

Pick a query from the CzechLynx test split and compare its top 5 under cosine retrieval,
default LoMa and LoMa + WildMatch, with gallery photos of the true individual framed in
blue. The filters select queries that fine-tuning rescued, still gets wrong, regressed on,
or already had right; the counts above the examples are over the whole test split, and the
examples themselves are a random sample of each category.

<div id="wm-rank-change" class="wm-widget">Loading the rank-change explorer…</div>

- **Rescued**: the fine-tuned matcher ranks the true individual first and the default
  matcher does not.
- **Regressed**: the opposite. Both counts matter; the paper reports the net effect.
- **Still wrong**: neither matcher finds the individual at rank 1. Many of these queries
  are night or infrared frames, or the individual is outside the candidate list.
- **Already right**: both matchers rank the true individual first.

All three methods rank the same 250 candidates per query, the paper's main budget. A dash
in the true-rank readout means the individual was not among them.

<small>Photographs from CzechLynx (Picek et al.), time-closed split.</small>

[Back to the demo overview](index.md){ .md-button }
