# Method

WildMatch adapts a pretrained local feature matcher to a wildlife domain using only the
identity labels of the reference database. Nothing in the procedure needs keypoint
annotations, geometric ground truth or camera calibration.

## Problem setup and inference

Each image \(x\) is processed by a frozen extractor into keypoints \(p_i\) and
descriptors \(z_i\). The matching module \(M_\theta\) takes the two keypoint sets of
an image pair and predicts a confidence matrix \(P \in [0,1]^{n_x \times n_{x'}}\).
Mutual-nearest pairs above a confidence threshold form the correspondence set
\(\mathcal{A}\), and the pair is scored by the summed confidence of its
correspondences, normalised by the smaller keypoint count:

\[
s(x, x') = \frac{1}{\min(n_x, n_{x'})} \sum_{(i,j) \in \mathcal{A}} P_{ij}.
\]

For a query, the matcher scores the \(k\) gallery images with the highest cosine
similarity of global embeddings and ranks these candidates by \(s\). All experiments use
MegaDescriptor-L for the candidate list, a fixed starting point shared by every
matching method and baseline; \(k\) is the candidate budget reported with each result.

**Background masking.** Pixels outside the animal mask are blacked out and nothing is
cropped, so every method sees the same inputs. CzechLynx ships its own masks, the
WildlifeReID-10k datasets provide pre-masked images, and SalamanderID2025 was segmented
for this work with text-prompted SAM 3 (prompt "Salamander", recorded in the dataset's
mask metadata).

!!! note "Draft"
    The manuscript's description of masking is being revised by the authors. The
    statement above follows the mask metadata and the segmentation script, not the
    current manuscript text.

## Identity-guided pair mining

Pairs are mined once, before training, with the pretrained matcher \(\theta_0\) on the
training (database) split, independently of the global retrieval stage.

- Images are subsampled and grouped into identity-specific collections.
- Each image serves as an anchor \(a\) and is matched against the other training images.
- The **positive pool** \(\mathcal{P}(a)\) holds the highest-scoring images of the same
  identity; the **negative pool** \(\mathcal{N}(a)\) holds the highest-scoring images of
  other identities, i.e. hard negatives. Each pool keeps up to five images.
- Anchors without a positive are dropped, and a limited number of anchors is kept per
  collection, ranked by their maximum pair score.

## Identity-supervised matcher adaptation

Only the matching module is updated. Detector, descriptor and the global encoder stay
frozen. For each anchor, one positive is sampled uniformly from \(\mathcal{P}(a)\) and
one negative from a mix of hard negatives in \(\mathcal{N}(a)\) and random images of
other identities.

Mutual selection and thresholding are not differentiable, so training uses a relaxed
score, the mean over keypoints of each keypoint's best assignment probability, averaged
over both directions:

\[
\tilde{s}(x, x') = \tfrac{1}{2}\Big( \operatorname{mean}_i \max_j P_{ij} + \operatorname{mean}_j \max_i P_{ij} \Big).
\]

The matching module is fine-tuned with a triplet margin objective, a contrastive loss
that encourages higher correspondence scores for same-identity pairs than for
different-identity pairs. It pushes the anchor-positive score above the anchor-negative
score by a margin \(\alpha = 0.5\):

\[
\mathcal{L} = \mathbb{E}\big[\, (\alpha - \tilde{s}(a,p) + \tilde{s}(a,n))_+ \,\big].
\]

### Implementation details

We fine-tune only the matching module (11.9M parameters) and keep the detector and
descriptor frozen. Inputs are 512-pixel images with up to 512 keypoints. Triplets are
mined once on the training images with the pretrained matcher. We train for 300 epochs
with AdamW, a batch size of 32, a learning rate of \(10^{-5}\) with a cosine schedule,
a weight decay of \(10^{-4}\), and gradient clipping at 1, and report the final
checkpoint. On CzechLynx, the training steps of fine-tuning take 5.1 GPU-hours on an NVIDIA
RTX 4090 (24 GB).

Inference ranking is unchanged by training: candidates are still scored with the
thresholded mutual-confidence score \(s\).

## Matchers and baselines

| Role | Model | Notes |
|---|---|---|
| Adapted matcher | LoMa | Matching module fine-tuned with WildMatch |
| Adapted matcher | RDD-LightGlue | RDD keypoints and descriptors with its released LightGlue matcher |
| Candidate list | MegaDescriptor-L | Cosine similarity of global embeddings; same \(k\) candidates for every matching method |
| Baseline | Off-the-shelf LoMa, RDD-LightGlue | Pretrained weights over the same candidates |
| Baseline | WildFusion | Calibrated fusion of global and local scores over the same candidates |
| Baseline | Cosine retrieval | MegaDescriptor-L or DINOv3-L embeddings, full gallery |
| Baseline | Identity classifier | Linear probe on MegaDescriptor-L with a class-weighted loss; backbone frozen, partially or fully fine-tuned |

Metrics are **top-5 accuracy** (primary) and **balanced top-1 accuracy**, the top-1
accuracy averaged over individuals.

!!! info "Caveat on global-embedding baselines"
    MegaDescriptor-L was trained on six of the eight datasets (all except Sea star,
    CzechLynx and Salamander). This favours the cosine and classifier baselines and
    shapes the candidate lists on those datasets.
