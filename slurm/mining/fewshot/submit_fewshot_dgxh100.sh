#!/usr/bin/env bash
# Submit one self-contained few-shot job per training-data fraction to the dgxh100 partition.
#
# The per-user limits there (QoS p_high_gpu: 6 GPUs, 60 CPUs, 384 GB) are shared by the
# jobs: with 1 GPU, 10 CPUs and 64 GB each, six of them run concurrently (a job actually
# uses ~5 GB RSS). Every job runs the whole chain (view -> mining -> aggregation ->
# fine-tuning -> evaluation) for its fraction, see fewshot_job.sh.
#
# usage (from the rdd-parallel-benchmark root):
#   bash slurm_scripts/fewshot/submit_fewshot_dgxh100.sh configs/wildlife/CowDataset.json [seed]
# environment:
#   FEWSHOT_FRACTIONS  ("0.125 0.25 0.5 1.0")
#   FEWSHOT_PARTITION  (dgxh100)  FEWSHOT_QOS (quick)  FEWSHOT_TIME (24:00:00)
#   FEWSHOT_GPUS (1)   FEWSHOT_CPUS (10)   FEWSHOT_MEM (64G)
#   plus everything fewshot_job.sh understands (FEWSHOT_BACKEND, FEWSHOT_FORCE, ...).
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
config=${1:?usage: $0 configs/wildlife/<dataset>.json [seed]}
seed=${2:-${FEWSHOT_SEED:-0}}
[[ -f "${config}" ]] || { echo "config not found: ${config}" >&2; exit 1; }
dataset=$(basename "${config}" .json)
fractions=${FEWSHOT_FRACTIONS:-0.125 0.25 0.5 1.0}
mkdir -p logs/fewshot

echo "dataset=${dataset} seed=${seed} backend=${FEWSHOT_BACKEND:-rdd} fractions=${fractions}"
echo "partition=${FEWSHOT_PARTITION:-dgxh100} qos=${FEWSHOT_QOS:-quick} gpus=${FEWSHOT_GPUS:-1} cpus=${FEWSHOT_CPUS:-10} mem=${FEWSHOT_MEM:-64G} time=${FEWSHOT_TIME:-24:00:00}"
for fraction in ${fractions}; do
  job=$(sbatch --parsable \
    --job-name="fs-${dataset}-${fraction}-${FEWSHOT_BACKEND:-rdd}" \
    --partition="${FEWSHOT_PARTITION:-dgxh100}" --qos="${FEWSHOT_QOS:-quick}" \
    --gres="gpu:${FEWSHOT_GPUS:-1}" --cpus-per-task="${FEWSHOT_CPUS:-10}" \
    --mem="${FEWSHOT_MEM:-64G}" --time="${FEWSHOT_TIME:-24:00:00}" \
    slurm_scripts/fewshot/fewshot_job.sh "${config}" "${fraction}" "${seed}")
  echo "fraction ${fraction}: job ${job}  (log: logs/fewshot/fs-${dataset}-${fraction}-${FEWSHOT_BACKEND:-rdd}-${job}.out)"
done
echo "watch: squeue -u \$USER -p ${FEWSHOT_PARTITION:-dgxh100}"
