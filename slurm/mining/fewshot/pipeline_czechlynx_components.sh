#!/usr/bin/env bash
# Descriptor fine-tuning on a CzechLynx few-shot view (default: fraction 1.0, seed 0): trains a
# keypoint-descriptor network instead of, or together with, the matcher, on the pairs that
# pipeline_czechlynx.sh mined with the same backend, then probes the checkpoint.
#
#   backend rdd:  component lg              LightGlue on cached RDD features (the matcher run of
#                                           pipeline_czechlynx.sh; --trained_model lg)
#                 component descriptor      RDD's descriptor network through the frozen LightGlue
#                                           (train_by_lg_matches --trained_model rdd
#                                            --rdd_train_component descriptor)
#                 component lg+descriptor   LightGlue and RDD's descriptor jointly (--trained_model lg+rdd)
#   backend loma: component matcher         the LoMa matcher on cached features (pipeline_czechlynx.sh's run)
#                 component descriptor      DeDoDe's descriptor through the frozen LoMa matcher
#                 component descriptor+matcher   both jointly
#
# FEWSHOT_EPOCHS (300) sets the training length (cosine schedule over it); a run with another
# length is a separate experiment and gets the tag ep<N> (FEWSHOT_TAG overrides) in its
# checkpoint directory, job/run names and probe labels, so it never collides with the
# 300-epoch runs: checkpoints/.../<view>/<backend>[-<component>]-finetuned[-<tag>]/, probe
# variants custom[-<component>][-<tag>], report series "... (fine-tuned descriptor, ep30)".
#
# The RDD detector / DaD detector stay frozen in every mode (NMS keypoints carry no gradient),
# so LoMa keeps its DaD keypoint cache; the descriptors themselves are recomputed with gradient
# every step, which is what makes these runs several times slower than the matcher runs
# (Turhan's descriptor runs on 4 x rtx4090: RDD ~8 min/epoch, LoMa ~6 min/epoch => 300
# epochs take 30-40 h). The rtx4090_batch QoS allows 24 h per job, so the training is submitted
# as FEWSHOT_TRAIN_CHAIN jobs chained with --dependency=afterany: every job resumes from the
# newest complete epoch of the previous one (czechlynx_train_job.sh) and exits at once when
# the final checkpoint exists.
#
# Steps: 1. the view and the backend's mined indices must exist (pipeline_czechlynx.sh); when
#           the indices are still being mined, the training waits for the queued aggregation
#           job fs-CzechLynx-<fraction>-<backend>-aggregate (--dependency=afterok);
#        2. training chain on rtx4090_batch, checkpoints under
#           $FEWSHOT_ROOT/checkpoints/CzechLynx/legacy/<view>/<backend>-<component>-finetuned/
#           ('+' in the component becomes '-': rdd-lg-descriptor-finetuned, loma-descriptor-matcher-finetuned);
#        3. a CPU chain job after the last training job runs probe-fewshot-wildlife.sh CzechLynx
#           (reduced + full gallery; the launcher picks up every descriptor checkpoint of the
#           view as the variant custom-<component>) and the collector
#           (reports/fewshot/CzechLynx/, series "RDD-LightGlue (fine-tuned descriptor)" etc.).
#
# usage (from the rdd-parallel-benchmark root):
#   bash slurm_scripts/fewshot/pipeline_czechlynx_components.sh <rdd|loma> <component> [fraction] [seed]
#   bash slurm_scripts/fewshot/pipeline_czechlynx_components.sh rdd descriptor
#   bash slurm_scripts/fewshot/pipeline_czechlynx_components.sh loma descriptor+matcher 1.0 0
# environment:
#   FEWSHOT_CZ_COMPONENT_SBATCH   training resources; default "-p rtx4090_batch --qos=batch --gres=gpu:4
#                                 --cpus-per-task=40 --mem=125G --time=1-00:00:00 --exclude=c11"
#                                 (one rtx4090 node, 4 x batch 8 = global 32 for LoMa, 4 x 4 = 16 for RDD)
#   FEWSHOT_TRAIN_CHAIN           chained 24 h jobs per training (extra ones exit immediately);
#                                 default 3 for more than 100 epochs, else 1 (and --time=12:00:00)
#   FEWSHOT_GLOBAL_BATCH          overrides the reference global batch (rdd 16, loma 32)
#   FEWSHOT_EPOCHS (300), FEWSHOT_EVAL_EPOCH (last epoch), FEWSHOT_EVAL_EVERY (10 for <= 100 epochs,
#                                 else the wrapper default), FEWSHOT_TAG (ep<N> when FEWSHOT_EPOCHS != 300)
#   FEWSHOT_SKIP_PROBE=1, FEWSHOT_DRY_RUN=1 (print the sbatch calls only)
#   PROBE_SBATCH_ARGS ("-p rtx4090_batch --qos=batch --exclude=c11,c15,c22"), PROBE_CANDIDATE_K ("50")
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
activate_conda_env "${CONDA_ENV_RDD}"
backend=${1:?usage: $0 <rdd|loma> <component> [fraction] [seed]}
component=${2:?usage: $0 <rdd|loma> <component> [fraction] [seed]}
fraction=${3:-1.0}
seed=${4:-${FEWSHOT_SEED:-0}}
dataset=CzechLynx
protocol=legacy
epochs=${FEWSHOT_EPOCHS:-300}
epoch=${FEWSHOT_EVAL_EPOCH:-$(( epochs - 1 ))}
if (( epochs > 100 )); then
  chain_length=${FEWSHOT_TRAIN_CHAIN:-3}; time_limit=1-00:00:00; eval_every=${FEWSHOT_EVAL_EVERY:-}
else
  chain_length=${FEWSHOT_TRAIN_CHAIN:-1}; time_limit=12:00:00; eval_every=${FEWSHOT_EVAL_EVERY:-10}
fi
tag=${FEWSHOT_TAG-$([[ "${epochs}" != 300 ]] && echo "ep${epochs}")}
case "${backend}:${component}" in
  rdd:lg|loma:matcher) matcher_run=1 ;;
  rdd:descriptor|rdd:lg+descriptor|rdd:rdd|rdd:lg+rdd) matcher_run=0 ;;
  loma:descriptor|loma:descriptor+matcher) matcher_run=0 ;;
  *) echo "unsupported backend/component ${backend}/${component}: rdd -> lg, descriptor, lg+descriptor (rdd, lg+rdd: detector unfrozen too); loma -> matcher, descriptor, descriptor+matcher" >&2; exit 1 ;;
esac
component_tag=${component//+/-}
train_sbatch=${FEWSHOT_CZ_COMPONENT_SBATCH:--p rtx4090_batch --qos=batch --gres=gpu:4 --cpus-per-task=40 --mem=125G --time=${time_limit} --exclude=c11}
[[ -d "${EXREID_ROOT}" && -d "${LYNX_FINETUNING_ROOT}" ]] || { echo "EXREID_ROOT / LYNX_FINETUNING_ROOT must exist" >&2; exit 1; }
if [[ -n "${FEWSHOT_DRY_RUN:-}" ]]; then sbatch() { echo "[dry-run] sbatch $*" >&2; echo 0; }; fi
mkdir -p logs/fewshot

view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${fraction}, ${seed}))")
view_root=${FEWSHOT_ROOT}/views/${dataset}/${protocol}/${view}
index_root=${FEWSHOT_ROOT}/indices/${dataset}/${protocol}/${view}/${backend}
if (( matcher_run )); then output_dir=${FEWSHOT_ROOT}/checkpoints/${dataset}/${protocol}/${view}/${backend}-finetuned${tag:+-${tag}}
else output_dir=${FEWSHOT_ROOT}/checkpoints/${dataset}/${protocol}/${view}/${backend}-${component_tag}-finetuned${tag:+-${tag}}; fi
run_name=${dataset}-${backend}-${component_tag}-${protocol}-${view}${tag:+-${tag}}
job_name=fs-${dataset}-${fraction}-${backend}-${component_tag}-train${tag:+-${tag}}
echo "== CzechLynx fine-tuning: backend ${backend}, component ${component}, ${epochs} epochs${tag:+ (tag ${tag})}, view ${view}"
[[ -f "${view_root}/fewshot.json" ]] || { echo "view ${view_root} does not exist; run pipeline_czechlynx.sh first" >&2; exit 1; }

# 1. mined indices of this backend (or the queued aggregation that produces them) ------------
train_dependency=""
if [[ -f "${index_root}/strong-matches_train_combined.json" && -f "${index_root}/strong-matches_test_combined.json" ]]; then
  echo "indices: ${index_root}"
else
  aggregate=$(squeue -h -u "${USER}" -n "fs-${dataset}-${fraction}-${backend}-aggregate" -o "%i" 2>/dev/null | head -n 1)
  if [[ -z "${aggregate}" ]]; then
    echo "no ${backend} indices under ${index_root} and no queued aggregation job; run" >&2
    echo "  FEWSHOT_BACKEND=${backend} bash slurm_scripts/fewshot/pipeline_czechlynx.sh ${seed}" >&2
    echo "first (it mines the pairs with ${backend}), then re-run this script" >&2
    exit 1
  fi
  train_dependency="--dependency=afterok:${aggregate}"
  echo "indices: not yet mined, waiting for aggregation job ${aggregate}"
fi

# 2. training chain ---------------------------------------------------------------------------
epoch_dir=$(printf 'epoch_%03d' "${epoch}"); [[ "${backend}" == rdd ]] && epoch_dir=$(printf 'epoch_%02d' "${epoch}")
train_jobs=()
if [[ -f "${output_dir}/${epoch_dir}/model.safetensors" && -z "${FEWSHOT_FORCE:-}" ]]; then
  echo "checkpoint exists: ${output_dir}/${epoch_dir} (FEWSHOT_FORCE=1 retrains)"
elif [[ -n "$(squeue -h -u "${USER}" -n "${job_name}" -o '%i' 2>/dev/null)" && -z "${FEWSHOT_FORCE:-}" ]]; then
  echo "training already queued/running: $(squeue -h -u "${USER}" -n "${job_name}" -o '%i' | tr '\n' ' ')"
elif (( matcher_run )) && [[ -z "${tag}" && -n "$(squeue -h -u "${USER}" -n "fs-${dataset}-${fraction}-${backend}-train" -o '%i' 2>/dev/null)" ]]; then
  echo "the matcher training of pipeline_czechlynx.sh is already queued/running for ${view}: $(squeue -h -u "${USER}" -n "fs-${dataset}-${fraction}-${backend}-train" -o '%i' | tr '\n' ' ')"
else
  dependency=${train_dependency}
  for (( link = 1; link <= chain_length; link++ )); do
    # shellcheck disable=SC2086
    job=$(sbatch --parsable ${dependency} ${train_sbatch} --job-name="${job_name}" \
      --export="ALL,FEWSHOT_EPOCHS=${epochs},FEWSHOT_EVAL_EPOCH=${epoch}${eval_every:+,FEWSHOT_EVAL_EVERY=${eval_every}}" \
      slurm_scripts/fewshot/czechlynx_train_job.sh "${view_root}" "${index_root}" "${output_dir}" "${run_name}" "${backend}" "${component}")
    train_jobs+=("${job}"); echo "   training job ${link}/${chain_length}: ${job}${dependency:+ (${dependency})}"
    dependency="--dependency=afterany:${job}"
  done
  echo "   checkpoints: ${output_dir}/  (W&B project czechlynx-fewshot-${backend}, run ${run_name})"
fi

# 3. probes + collection ----------------------------------------------------------------------
if [[ -n "${FEWSHOT_SKIP_PROBE:-}" ]]; then echo "FEWSHOT_SKIP_PROBE set: no probe chain (training jobs: ${train_jobs[*]:-none})"; exit 0; fi
dependency=""
if (( ${#train_jobs[@]} > 0 )); then dependency="--dependency=afterany:${train_jobs[-1]}"; fi
# shellcheck disable=SC2086
chain=$(cd "${EXREID_ROOT}" && mkdir -p logs/fewshot && \
  FEWSHOT_SEED=${seed} FEWSHOT_PROTOCOL=${protocol} FEWSHOT_EVAL_EPOCH=${epoch} PROBE_FEWSHOT_TAG=${tag} \
  sbatch --parsable ${dependency} --job-name="fs-probe-${dataset}-${backend}-${component_tag}${tag:+-${tag}}" slurm/fewshot_probe_chain.sh "${dataset}" "${fraction}")
echo "probe chain job ${chain}${dependency:+ (${dependency})}: probes every missing variant of ${view} (reduced + full gallery), then the collector"
echo "results: ${EXREID_ROOT}/reports/fewshot/${dataset}/"
[[ -n "${FEWSHOT_DRY_RUN:-}" ]] && exit 0
record_dir=${FEWSHOT_ROOT}/runs/${dataset}/${protocol}
mkdir -p "${record_dir}"
cat > "${record_dir}/pipeline-components-${backend}-${component_tag}${tag:+-${tag}}-${view}-$(date -u +%Y%m%dT%H%M%SZ).json" <<EOF
{"dataset": "${dataset}", "seed": ${seed}, "backend": "${backend}", "component": "${component}", "protocol": "${protocol}",
 "epochs": ${epochs}, "tag": "${tag}", "fraction": "${fraction}", "train_jobs": "${train_jobs[*]:-}", "probe_chain_job": "${chain}",
 "output_dir": "${output_dir}", "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
