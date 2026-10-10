---
title: WildMatch
emoji: 🐆
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
suggested_hardware: zero-a10g
license: cc-by-nc-4.0
short_description: Match individual animals by their markings
models:
  - turhancan97/wildmatch-checkpoints
  - BVRA/MegaDescriptor-L-384
  - facebook/sam3
tags:
  - wildlife
  - re-identification
  - feature-matching
---

# WildMatch demo

WildMatch adapts pretrained feature matchers (LoMa, RDD-LightGlue) to individual animals using only identity
labels. This Space shows what the adapted matchers match:

- **Match two photos**: upload two photos, or pick an example pair (a lynx by day and by night, then one
  individual each of leopard, hyena, whale shark, sea star, sea turtle and fire salamander). Pick the species and
  compare the default matcher with its fine-tuned WildMatch checkpoint. Lines show strong, spread-out correspondences.
- **Find this lynx**: one lynx photo against a gallery of 160 photos of 44 lynx from the paper's unseen-individual
  test (none of them was in the fine-tuning data), or against synthetic lynx renders.

SAM 3 removes the background first, as in the paper: matching runs on the masked photos, and the lines are drawn
on the originals.

Paper: [arXiv:2610.07384](https://arxiv.org/abs/2610.07384) · Project page: <https://wildmatch.gmum.net> ·
Code: <https://github.com/turhancan97/WildMatch> · Checkpoints:
[turhancan97/wildmatch-checkpoints](https://huggingface.co/turhancan97/wildmatch-checkpoints)

## Licences and credits

- Fine-tuned checkpoints: CC BY-NC 4.0, non-commercial use only. MegaDescriptor-L (candidate list): CC BY-NC 4.0.
- Background removal: SAM 3 (Meta), under the SAM License.
- Gallery and lynx example photos: CzechLynx dataset (Picek et al.; Kaggle `picekl/czechlynx`) and its synthetic
  subset (Zenodo record 17592004), both CC BY 4.0; see `gallery/*/ATTRIBUTION.md` and `examples/ATTRIBUTION.md`.
- One example pair for each other species, taken from each dataset's original release: Leopard ID 2022 and Hyena ID
  2022 (Botswana Predator Conservation Trust, African Carnivore Wildbook), Whale Shark ID (Holmberg, Norman and
  Arzoumanian 2009) and Sea Star Re-ID 2023 (Wahltinez and Wahltinez 2024), all via LILA BC under
  CDLA-Permissive-1.0; Turtle Recall (Zindi, Local Ocean Conservation, Google DeepMind) under CC BY-SA 4.0; two
  SalamanderID2025 photos (AnimalCLEF 2025) shown with the organisers' permission, non-commercial. Cropped and
  resized; see `examples/species/ATTRIBUTION.md`.
- Code: Apache-2.0, The WildMatch Authors.

## Privacy

Uploaded photos are processed in memory and are not stored or logged by this Space.

## Citation

```bibtex
@misc{kargin2026wildmatch,
  title={WildMatch: Weakly Supervised Image Matcher Adaptation for Wildlife Re-Identification},
  author={Turhan Can Kargin and Piotr Kubaty and Ekaterina Rostovskaya and Izabela Wierzbowska and Bartosz Zieliński and Marcin Przewięźlikowski},
  year={2026},
  eprint={2610.07384},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.07384}
}
```

## Running it elsewhere

The source lives in the WildMatch repository under `space/`; `paper/tools/deploy_space.py` copies the `wildmatch`
package next to `app.py` and uploads the folder. Locally: `pip install -r requirements.txt gradio`, then
`python app.py` (`HF_TOKEN` of an account with access to `facebook/sam3`, or `WILDMATCH_SAM3_CHECKPOINT`).
