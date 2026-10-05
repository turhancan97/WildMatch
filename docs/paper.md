# Paper & Code

**WildMatch: Weakly Supervised Image Matcher Adaptation for Wildlife
Re-Identification.** The manuscript is not yet publicly available; a preprint link will
appear here when it is online.

## Citation

```bibtex
% BibTeX entry to be provided by the authors once the preprint is online.
```

## Acknowledgements

!!! note "To be completed"
    Funding sources, computing grants and data providers to be acknowledged will be
    listed here by the authors.

## Contact

Questions about the method or the code: open an issue in the evaluation repository
linked below.

## Code

One repository implements the whole pipeline: pair mining, matcher fine-tuning and evaluation.

| Repository | Role | Link |
|---|---|---|
| `WildMatch` | Pair mining, matcher fine-tuning (LoMa, RDD-LightGlue), evaluation: probes, candidate lists, metrics, run manifests, paper tables and figures, this page | [GitHub](https://github.com/turhancan97/WildMatch) |

### Repository map

- `src/wildmatch/`: the installable package behind the `wildmatch` command.
- `src/wildmatch/evaluate/`: probe dispatch (cosine, WildFusion, Vismatch matchers, linear and
  efficient probes), metrics (top-k, balanced top-1, mAP) and stable ranking.
- `src/wildmatch/matchers/`: Vismatch matcher profiles and custom-checkpoint loading.
- `src/wildmatch/mining/`: pair mining (`wildmatch mine`): dataset views, feature caches,
  strong-match indices.
- `src/wildmatch/matcher_finetune/`: matcher fine-tuning (`wildmatch finetune-matcher`): triplet
  sampling, relaxed score, training recipe, checkpoints.
- `src/wildmatch/data/`: dataset views, COCO-RLE masking, split safety checks and dataset
  preparation (`wildmatch prepare`).
- `src/wildmatch/sweep/`: evaluation grids as immutable Slurm or local submissions.
- `src/wildmatch/reporting/`: run manifests, run index, paper tables and figures.
- `src/wildmatch/conf/`: Hydra configuration, path profiles and the dataset registry.
- `paper/`: page exporters and paper figures.
- `tests/`: dependency-light regression tests.

### Third-party components

| Component | Use | Pinning |
|---|---|---|
| MegaDescriptor-L | Global embeddings for the candidate list and classifier baselines | wildlife-tools release |
| DINOv3-L | Cosine-retrieval baseline | `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated) |
| LoMa | Matcher | via Vismatch, pinned commit (evaluation); `lomatch`, pinned commit (mining, fine-tuning) |
| RDD, LightGlue | Matcher | via Vismatch, pinned commit (evaluation); RDD code vendored unchanged (mining, fine-tuning) |
| WildFusion | Baseline | wildlife-tools |
| SAM 3 | Background removal for SalamanderID2025 | prompt "Salamander" |

Licenses of wrapped models and downloaded weights are being collected for the release.
