#!/usr/bin/env bash
# Shared path/parameter resolution for the few-shot scripts. Sourced (not run) after
# env.sh has been loaded, the conda env activated and the cwd set to the benchmark root:
#
#   fewshot_resolve <config> <fraction> <seed>
#
# reads FEWSHOT_BACKEND (rdd|loma, default rdd) and WILDLIFE_PROTOCOL (default legacy) and
# exports: config dataset protocol backend fraction seed view view_root index_root
# ckpt_root eval_root run_record cache pretrained train_script train_log_dir output_dir
# eval_epoch finetuned, plus WILDLIFE_CONFIG/WILDLIFE_PROTOCOL/WILDLIFE_DATASET_ID.

fewshot_resolve() {
  local config_arg=${1:?config} fraction_arg=${2:?fraction} seed_arg=${3:-0}
  config=$(readlink -f "${config_arg}")
  [[ -f "${config}" ]] || { echo "config not found: ${config_arg}" >&2; return 1; }
  fraction=${fraction_arg}
  seed=${seed_arg}
  backend=${FEWSHOT_BACKEND:-rdd}
  protocol=${WILDLIFE_PROTOCOL:-legacy}
  case "${backend}" in rdd|loma) ;; *) echo "FEWSHOT_BACKEND must be rdd or loma (got ${backend})" >&2; return 1 ;; esac
  eval "$(python -m scripts.wildlife_config --config "${config}" --shell)"
  view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${fraction}, ${seed}))")
  dataset=${WILDLIFE_DATASET_ID}
  view_root=${FEWSHOT_ROOT}/views/${dataset}/${protocol}/${view}
  index_root=${FEWSHOT_ROOT}/indices/${dataset}/${protocol}/${view}
  ckpt_root=${FEWSHOT_ROOT}/checkpoints/${dataset}/${protocol}/${view}
  eval_root=${FEWSHOT_ROOT}/eval/${dataset}/${protocol}/${view}
  run_record=${FEWSHOT_ROOT}/runs/${dataset}/${protocol}/${view}-${backend}.json
  if [[ "${backend}" == rdd ]]; then
    cache=${WILDLIFE_RDD_CACHE:-${CHECKPOINTS_ROOT}/wildlife-reid-10k/${dataset}/rdd-cache}
    pretrained=${LG_WEIGHTS}
    train_script=slurm_scripts/train_wildlife_rdd.sh
    train_log_dir=logs/wildlife-rdd-ft
  else
    cache=${WILDLIFE_LOMA_CACHE:-${CHECKPOINTS_ROOT}/wildlife-reid-10k/${dataset}/loma-cache}
    pretrained=${LOMA_WEIGHTS}
    train_script=slurm_scripts/train_wildlife_loma.sh
    train_log_dir=logs/wildlife-loma-ft
  fi
  [[ -f "${cache}/manifest.json" ]] || { echo "${backend} cache not found: ${cache}" >&2; return 1; }
  [[ -e "${pretrained}" ]] || { echo "pretrained ${backend} checkpoint not found: ${pretrained}" >&2; return 1; }
  [[ -d "${LYNX_FINETUNING_ROOT}" ]] || { echo "LYNX_FINETUNING_ROOT does not exist: ${LYNX_FINETUNING_ROOT}" >&2; return 1; }
  output_dir=${ckpt_root}/${backend}-finetuned
  eval_epoch=${FEWSHOT_EVAL_EPOCH:-299}
  finetuned=${output_dir}/epoch_${eval_epoch}/model.safetensors
  export WILDLIFE_CONFIG=${config} WILDLIFE_PROTOCOL=${protocol}
  export config dataset protocol backend fraction seed view view_root index_root ckpt_root \
    eval_root run_record cache pretrained train_script train_log_dir output_dir eval_epoch finetuned
}

# Export the trainer's environment (train_wildlife_{rdd,loma}.sh) for the resolved run.
# Call inside a subshell: ( fewshot_export_train_env; cd "$LYNX_FINETUNING_ROOT"; bash ... ).
fewshot_export_train_env() {
  export WILDLIFE_CONFIG=${config} WILDLIFE_PROTOCOL=${protocol}
  export WILDLIFE_VIEW_ROOT=${view_root} WILDLIFE_INDEX_ROOT=${index_root}/${backend}
  export WILDLIFE_RDD_CACHE=${cache} WILDLIFE_LOMA_CACHE=${cache}
  export WILDLIFE_RDD_OUTPUT=${output_dir} WILDLIFE_LOMA_OUTPUT=${output_dir}
  export WILDLIFE_RDD_RUN_NAME=${dataset}-${backend}-${protocol}-${view}
  export WILDLIFE_LOMA_RUN_NAME=${dataset}-${backend}-${protocol}-${view}
  export WILDLIFE_WANDB_PROJECT=${WILDLIFE_WANDB_PROJECT:-wildlife-fewshot-${backend}-${dataset}}
}

# Export wildlife_evaluate.sh's environment: fewshot_export_eval_env <weights> <output-json>
fewshot_export_eval_env() {
  export WILDLIFE_CONFIG=${config} WILDLIFE_PROTOCOL=${protocol} WILDLIFE_VIEW_ROOT=${view_root}
  export WILDLIFE_BACKEND=${backend} WILDLIFE_CACHE=${cache} WILDLIFE_MODE=full
  export WILDLIFE_WEIGHTS=${1:?weights} WILDLIFE_OUTPUT=${2:?output}
}

fewshot_write_record() {
  # fewshot_write_record <launcher> <json fragment with job ids or step timings>
  mkdir -p "$(dirname "${run_record}")"
  cat > "${run_record}" <<EOF
{
  "dataset": "${dataset}", "protocol": "${protocol}", "view": "${view}",
  "fraction": ${fraction}, "seed": ${seed}, "backend": "${backend}", "launcher": "${1:?launcher}",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "view_root": "${view_root}", "index_root": "${index_root}/${backend}",
  "checkpoint_dir": "${output_dir}", "eval_root": "${eval_root}",
  "cache": "${cache}", "pretrained": "${pretrained}", "finetuned": "${finetuned}",
  ${2:?fragment}
}
EOF
  echo "run record: ${run_record}"
}
