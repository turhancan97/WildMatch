# Third-party licences

The WildMatch code is Apache-2.0 (`LICENSE`). The software and models it uses keep their own
licences. Checked on 2026-10-04 from the installed packages' metadata, the upstream repositories'
`LICENSE` files and the model cards; re-check before a release.

## Code dependencies

| Component | Licence | Source of the check |
| --- | --- | --- |
| Vismatch (matcher wrappers) | BSD-3-Clause | `LICENSE` of gmberton/vismatch; wrapped matchers keep their own licences |
| LoMa (code) | MIT, except the matcher, which inherits Apache-2.0 from LightGlue | `LICENSE` and README of davnords/LoMa |
| RDD (code) | Apache-2.0 | `LICENSE` of xtcpete/rdd (fork turhancan97/rdd); a subset is vendored unchanged in `src/wildmatch/vendor/rdd/` with its `LICENSE` (see `VENDORED.md`) |
| LightGlue, modified copies | Apache-2.0 | `src/wildmatch/matcher_finetune/rdd_patch/` and `src/wildmatch/mining/lightglue_masked.py` are modified from cvg/LightGlue (via RDD), as their headers state |
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
| RDD and RDD-LightGlue weights (`RDD-v2.pth`, `RDD_lg-v2.pth`) | Apache-2.0 | published in the xtcpete/rdd repository, whose licence is Apache-2.0 |
| LoMa-B weights (`loma_B.pt`) | no weight licence stated; read as non-commercial | checked 2026-10-06: the davnords/LoMa README and PyPI page state licences for the code only. The weights are release assets of `davnords/storage`, a repository whose licence file is MIT, but nothing says that licence covers the assets. LoMa was trained on 17 datasets, among them ScanNet++ (non-commercial research only), Niantic Map-free (non-commercial) and CO3Dv2 (CC BY-NC 4.0), and on 2026-09-26 the author added separate, undocumented `commercial_loma_*` weights next to the original `loma_B.pt` (2026-03-28), which suggests the original is meant for non-commercial use. Our checkpoints were fine-tuned from the original (757,888,113 bytes, SHA-256 `3a388243...`). Written confirmation from the authors is optional |
| WildMatch fine-tuned matcher checkpoints (`wildmatch weights`) | **CC BY-NC 4.0** (user decision 2026-10-06) | fine-tuned from the weights above on data with non-commercial terms (WildlifeReID-10k); non-commercial like MegaDescriptor-L, which every method uses for the candidate list |
| SAM 3 | SAM License | redistribution of SAM Materials or derivatives must follow the agreement; publications of research using them must acknowledge SAM |

## Datasets

Datasets are not redistributed here; `wildmatch prepare` downloads or rebuilds them from their
published sources. Derived data is not redistributed either: the SAM 3 masks for
WildlifeReID-10k are released as a recipe (`wildmatch prepare` and `slurm/sam3_masks.sbatch`),
not as mask files (user decision 2026-10-06), because the WildlifeReID-10k terms forbid
re-uploading. Each registry entry records its licence in `registry.licence`. Checked on
2026-10-06 from the Kaggle dataset metadata (through the Kaggle API) and the source pages:

| Dataset | Licence | Notes |
| --- | --- | --- |
| WildlifeReID-10k (Kaggle `wildlifedatasets/wildlifereid-10k`) | "Other": no commercial use, no re-upload, attribution of WildlifeReID-10k and of every source dataset | applies on top of each source licence below |
| HyenaID2022, LeopardID2022, SeaStarReID2023, WhaleSharkID, BelugaID, GiraffeZebraID | CDLA-Permissive-1.0 | via WildlifeReID-10k; the LeopardID2022 authors also confirmed by email (2026-10-07) that showing their photos is fine for research purposes |
| ZindiTurtleRecall | CC BY-SA 4.0 | via WildlifeReID-10k |
| NyalaData, Giraffes | none stated by the source | via WildlifeReID-10k; strictly all rights reserved |
| ATRW | CC BY-NC-SA 4.0 | via WildlifeReID-10k |
| StripeSpotter | CC BY-SA 3.0 | via WildlifeReID-10k |
| CowDataset | CC BY 4.0 | via WildlifeReID-10k |
| CzechLynx (Kaggle `picekl/czechlynx`) | CC BY 4.0 | also covers the synthetic renders bundled with `wildmatch demo` |
| SalamanderID2025 (AnimalCLEF2025, Kaggle competition) | competition rules: non-commercial use, including academic research and education | rules read 2026-10-06 (section 4a). The data must not be published or redistributed to anyone outside the competition (4b), so showing photos needed the organisers' permission: granted by email on 2026-10-07 (no objection to showing some photos in the repository or on the project page). The SeaTurtleID2022 clause covers the turtle part only, which we do not use |
| JaguarReID (Kaggle Jaguar Re-ID competition) | competition use only | rules read 2026-10-06 (section 2.4): research, publications, training models outside the competition and derivative datasets need written authorization from the sponsors (Antonio Rueda-Toicen, Abigail Martin), and the data had to be deleted within 30 days after the competition ended (2026-03-17) unless authorized. Our 2026-10 use (the burst-aware split, the probe runs) is not covered. Authorization being requested (user decision 2026-10-06); no Jaguar runs until it is granted. Not a paper dataset, nothing on the page or the Hub |
