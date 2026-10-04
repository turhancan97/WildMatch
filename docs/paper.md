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

Three repositories implement the pipeline.

| Repository | Role | Link |
|---|---|---|
| `explainable_individual_reidentification` | Evaluation: probes, candidate lists, metrics, run manifests, paper tables and figures, this page | [GitHub](https://github.com/turhancan97/explainable_individual_reidentification) |
| `lynx-finetuning` | Matcher fine-tuning (LoMa, RDD-LightGlue): triplet sampling, relaxed score, training recipe, checkpoints | [GitHub](https://github.com/PiotrKubaty/lynx-finetuning) |
| `rdd-parallel-benchmark` | Pair mining: feature caches, strong-match indices, Slurm orchestration | [GitHub](https://github.com/PiotrKubaty/rdd-parallel-benchmark) |

!!! note
    Repository names and links are provisional; the evaluation repository is to be
    renamed before the page is published, and the links above will follow.

### Evaluation repository map

- `src/wildmatch/`: the installable package behind the `wildmatch` command.
- `src/wildmatch/evaluate/`: probe dispatch (cosine, WildFusion, Vismatch matchers, linear and
  efficient probes), metrics (top-k, balanced top-1, mAP) and stable ranking.
- `src/wildmatch/matchers/`: Vismatch matcher profiles and custom-checkpoint loading.
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
| LoMa | Matcher | via Vismatch, pinned commit |
| RDD, LightGlue | Matcher | via Vismatch, pinned commit |
| WildFusion | Baseline | wildlife-tools |
| SAM 3 | Background removal for SalamanderID2025 | prompt "Salamander" |

Licenses of wrapped models and downloaded weights are being collected for the release.
