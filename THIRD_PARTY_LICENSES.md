# Third-party licences

The WildMatch code is Apache-2.0 (`LICENSE`). The software and models it uses keep their own
licences. Checked on 2026-10-04 from the installed packages' metadata, the upstream repositories'
`LICENSE` files and the model cards; re-check before a release.

## Code dependencies

| Component | Licence | Source of the check |
| --- | --- | --- |
| Vismatch (matcher wrappers) | BSD-3-Clause | `LICENSE` of gmberton/vismatch; wrapped matchers keep their own licences |
| LoMa (code) | MIT, except the matcher, which inherits Apache-2.0 from LightGlue | `LICENSE` and README of davnords/LoMa |
| RDD (code) | Apache-2.0 | `LICENSE` of xtcpete/rdd (fork turhancan97/rdd) |
| LightGlue, gluefactory | Apache-2.0 | package metadata |
| wildlife-tools, wildlife-datasets | MIT | package metadata |
| PyTorch, torchvision, PoseLib | BSD-3-Clause | package metadata |
| timm, Transformers, Kornia | Apache-2.0 | package metadata |
| Hydra | MIT | package metadata |
| OmegaConf | BSD | package metadata |
| pycocotools | FreeBSD (BSD-2-Clause) | package metadata |
| SAM 3 (masking, separate environment) | SAM License (Meta) | `LICENSE` of the SAM 3 repository |

## Models and weights

| Model | Licence | Notes |
| --- | --- | --- |
| MegaDescriptor-L-384 (BVRA) | CC BY-NC 4.0 | model card on Hugging Face; non-commercial use only |
| LoMa-B and RDD-LightGlue default weights | as published by their authors | downloaded by Vismatch on first use; the code licences above do not state a separate weight licence (to confirm with the authors before a release) |
| WildMatch fine-tuned matcher checkpoints (`wildmatch weights`) | to decide | fine-tuned from the LoMa-B and RDD-LightGlue weights above; their release licence must be compatible with those |
| SAM 3 | SAM License | redistribution of SAM Materials or derivatives must follow the agreement; publications of research using them must acknowledge SAM |

## Datasets

Datasets are not redistributed here; `wildmatch prepare` downloads or rebuilds them from their
published sources. Each dataset's licence (WildlifeReID-10k and its sub-datasets, CzechLynx,
AnimalCLEF2025 SalamanderID2025, Kaggle Jaguar Re-ID) is still to be checked and recorded in the
registry (`registry.licence`).
