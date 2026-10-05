# Matcher fine-tuning

Command: `wildmatch finetune-matcher`. One recipe is shared by LoMa and RDD-LightGlue; only the
network being adapted differs.

## Recipe

| Element | Setting |
|---|---|
| Trained part | matching module only (LoMa matcher, or LightGlue for RDD); detector, descriptor and global encoder frozen |
| Triplets | one positive sampled uniformly from the anchor's positive pool; one negative from the hard-negative pool or, with probability 0.3, a random image of another individual |
| Training score | relaxed score: mean over each image's real keypoints of the best assignment probability, averaged over both directions, computed before mutual selection and thresholding |
| Loss | triplet margin, \((0.5 - \tilde{s}(a,p) + \tilde{s}(a,n))_+\), over every triplet |
| Optimiser | AdamW, learning rate \(10^{-5}\), weight decay \(10^{-4}\), cosine schedule, gradient clipping at 1 |
| Batch | effective batch 32: 8 per GPU on 4 GPUs (Salamander LoMa: 16 per GPU on 2); RDD descriptor and joint modes load 1 per GPU and accumulate 8 |
| Schedule | 300 epochs; the final checkpoint (epoch 299) is reported, no epoch selection |
| Validation score | the inference score (mutual matches above the confidence threshold), on the test index in the legacy protocol |
| Padding | RDD batches pad keypoints to the longest image; padded keypoints are excluded from the assignment, so they cannot be matched |

On CzechLynx the training steps of the LoMa run take 5.1 GPU-hours on RTX 4090 GPUs; the
RDD-LightGlue run takes 5.0.

## Training

`wildmatch finetune-matcher` takes the dataset's registry key and the matcher recipe
(`matcher_finetune=loma` or `rdd`); the view, the mined index, the feature cache and the output
folder follow from the dataset and the path profile. Every recipe value can be overridden on the
command line. `matcher_finetune.dry_run=true` prints the planned paths and the full training
command without running it:

```bash
wildmatch finetune-matcher dataset=nyala matcher_finetune=rdd matcher_finetune.dry_run=true
```

The Slurm script requests the four GPUs of the recipe:

```bash
# CzechLynx, time-closed split, legacy protocol, LoMa-mined pairs (defaults)
sbatch slurm/finetune_matcher.sbatch dataset=czechlynx_closed matcher_finetune=loma
sbatch slurm/finetune_matcher.sbatch dataset=czechlynx_closed matcher_finetune=rdd

# A WildlifeReID-10k dataset or Salamander, on the pairs its paper run used
sbatch slurm/finetune_matcher.sbatch dataset=nyala matcher_finetune=rdd matcher_finetune.mined_by=rdd
```

`matcher_finetune.mined_by` selects the pair source (`loma` by default; the source of each paper
run is listed under [Pair mining](mining.md#pair-sources-used-in-the-paper)) and
`matcher_finetune.protocol` the split protocol (`legacy`, the paper's). A run refuses to start in
an output directory that already holds epochs; `matcher_finetune.resume=auto` continues from the
newest complete epoch. The RDD trainer rejects a resume checkpoint whose objective, optimiser or
batch configuration differs; the LoMa trainer one that trained another component. RDD runs
and the CzechLynx LoMa runs keep every 50th epoch plus the first and the last
(`matcher_finetune.keep_every`).

## Checkpoint layout

```text
<checkpoint_root>/<dataset>/<matcher>-finetuned-.../
    czechlynx_protocol.json | wildlife_protocol.json   # split, backend, indices, component, objective
    wildmatch_provenance.json                          # one record per launch (see below)
    epoch_000/ epoch_050/ ... epoch_299/
        model.safetensors                              # LightGlue (RDD) or the LoMa bundle
        optimizer / scheduler / RNG state              # never loaded for evaluation
```

The protocol file records the trained component (`lg` or `matcher` for the paper's main
runs), the pair source, the split protocol and the training objective, and the evaluation
code validates it before loading a checkpoint. `wildmatch_provenance.json` records, for every
launch, the code commit and any uncommitted changes, the command, the seed, the checksums of
the indices, the pretrained weights and the feature cache, and the package versions.

The published checkpoints keep the folder names of the paper's runs; `wildmatch weights
download` places them where the evaluation looks for them (see [Evaluation](probing.md)).

## Reproducibility

The paper's checkpoints were trained with an earlier version of this training code, in a
separate environment with a different PyTorch version. Evaluating the published checkpoints with
the current package gives bit-identical scores; this was checked for all of them except the two
CzechLynx LoMa checkpoints, whose check did not fit in memory. Retraining with `wildmatch finetune-matcher` follows
the same recipe and gives equivalent checkpoints, not bit-identical ones: the PyTorch version
and nondeterministic GPU kernels change the trained weights slightly. The paper's numbers come
from the published checkpoints.

## Ablation variants

The "what to adapt" comparison trains the other parts with the same pairs and objective:

```bash
# descriptor branch only (detector and matcher frozen)
sbatch slurm/finetune_matcher.sbatch dataset=czechlynx_closed matcher_finetune=rdd matcher_finetune.component=descriptor
sbatch slurm/finetune_matcher.sbatch dataset=czechlynx_closed matcher_finetune=loma matcher_finetune.component=descriptor

# descriptor and matcher together (detector frozen)
sbatch slurm/finetune_matcher.sbatch dataset=czechlynx_closed matcher_finetune=rdd matcher_finetune.component=joint
sbatch slurm/finetune_matcher.sbatch dataset=czechlynx_closed matcher_finetune=loma matcher_finetune.component=joint
```

A trainable descriptor rules out the fixed feature cache, so features are recomputed every
step and these runs cost 16 to 21 times more than the matcher-only run. They are written to
separate `*-descriptor-finetuned-*` and `*-joint-finetuned-*` directories and are reported
separately from matcher-only fine-tuning.

## Unseen-identity protocol

The matcher for the unseen-identity experiment is trained the same way on the training
part of the CzechLynx time-open split (275 individuals), with pairs mined on that split:

```bash
sbatch slurm/finetune_matcher.sbatch dataset=czechlynx_open matcher_finetune=loma
```
