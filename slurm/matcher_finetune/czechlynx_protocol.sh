#!/usr/bin/env bash

# Resolve the official CzechLynx split column and training-time protocol.
czechlynx_resolve_split() {
    local split_column="${CZECHLYNX_SPLIT_COLUMN:-split-time_closed}"
    case "${split_column}" in
        split-time_closed|split-time_open) ;;
        *)
            echo "CZECHLYNX_SPLIT_COLUMN must be 'split-time_closed' or 'split-time_open', got: ${split_column}" >&2
            return 2
            ;;
    esac
    CZECHLYNX_RESOLVED_SPLIT_COLUMN="${split_column}"
    CZECHLYNX_RESOLVED_SPLIT_SUFFIX="${split_column#split-}"
    CZECHLYNX_RESOLVED_EXPERIMENT="czechlynx-${CZECHLYNX_RESOLVED_SPLIT_SUFFIX//_/-}"
}

# Resolve the CzechLynx index used for training-time validation.
# The legacy protocol mirrors the original Lynx workflow: test queries are
# used for validation, while strict uses the dedicated CzechLynx holdout.
czechlynx_resolve_protocol() {
    czechlynx_resolve_split || return $?
    local protocol="${CZECHLYNX_SPLIT_PROTOCOL:-legacy}"
    local backend="${CZECHLYNX_MINING_BACKEND:-rdd}"
    case "${backend}" in
        rdd|loma) ;;
        *) echo "CZECHLYNX_MINING_BACKEND must be 'rdd' or 'loma', got: ${backend}" >&2; return 2 ;;
    esac
    case "${protocol}" in
        legacy|strict) ;;
        *)
            echo "CZECHLYNX_SPLIT_PROTOCOL must be 'legacy' or 'strict', got: ${protocol}" >&2
            return 2
            ;;
    esac

    local experiment_root="/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/${CZECHLYNX_RESOLVED_EXPERIMENT}/${protocol}"
    local backend_root="${experiment_root}/${backend}"
    local index_root
    if [[ -n "${CZECHLYNX_INDEX_ROOT:-}" ]]; then
        # Explicit overrides retain their exact historical meaning.
        index_root="${CZECHLYNX_INDEX_ROOT}"
    elif [[ "${backend}" == "rdd" \
        && -f "${experiment_root}/strong-matches_train_combined.json" \
        && ! -e "${backend_root}" ]]; then
        # Read old RDD mining results that predate backend-specific folders.
        index_root="${experiment_root}"
    else
        index_root="${backend_root}"
    fi
    if [[ "${protocol}" == legacy ]]; then
        CZECHLYNX_RESOLVED_VAL_INDEX="${index_root}/strong-matches_test_combined.json"
        CZECHLYNX_RESOLVED_OUTPUT_SUFFIX="legacy"
    else
        CZECHLYNX_RESOLVED_VAL_INDEX="${index_root}/strong-matches_val_combined.json"
        CZECHLYNX_RESOLVED_OUTPUT_SUFFIX="strict"
    fi

    CZECHLYNX_RESOLVED_PROTOCOL="${protocol}"
    CZECHLYNX_RESOLVED_BACKEND="${backend}"
    CZECHLYNX_RESOLVED_INDEX_ROOT="${index_root}"
    CZECHLYNX_RESOLVED_TRAIN_INDEX="${CZECHLYNX_TRAIN_INDEX:-${index_root}/strong-matches_train_combined.json}"
    if [[ -n "${CZECHLYNX_VAL_INDEX:-}" ]]; then
        CZECHLYNX_RESOLVED_VAL_INDEX="${CZECHLYNX_VAL_INDEX}"
    fi
}
