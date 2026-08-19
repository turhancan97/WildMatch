from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "slurm_scripts" / "czechlynx_protocol.sh"


def resolve(protocol: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if protocol is None:
        env.pop("CZECHLYNX_SPLIT_PROTOCOL", None)
    else:
        env["CZECHLYNX_SPLIT_PROTOCOL"] = protocol
    command = (
        f"source {HELPER!s}; "
        "czechlynx_resolve_protocol; "
        "printf '%s\\n' "
        '"${CZECHLYNX_RESOLVED_PROTOCOL}" '
        '"${CZECHLYNX_RESOLVED_TRAIN_INDEX}" '
        '"${CZECHLYNX_RESOLVED_VAL_INDEX}" '
        '"${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}"'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    protocol_value, train_index, val_index, suffix = result.stdout.splitlines()
    return {
        "protocol": protocol_value,
        "train_index": train_index,
        "val_index": val_index,
        "suffix": suffix,
    }


def test_legacy_is_the_default_and_uses_test_index():
    result = resolve(None)
    assert result["protocol"] == "legacy"
    assert result["suffix"] == "legacy"
    assert result["train_index"].endswith("strong-matches_train_combined.json")
    assert result["val_index"].endswith("strong-matches_test_combined.json")


def test_strict_uses_the_dedicated_validation_index():
    result = resolve("strict")
    assert result["protocol"] == "strict"
    assert result["suffix"] == "strict"
    assert result["val_index"].endswith("strong-matches_val_combined.json")


def test_explicit_validation_override_wins():
    env = os.environ.copy()
    env["CZECHLYNX_SPLIT_PROTOCOL"] = "legacy"
    env["CZECHLYNX_VAL_INDEX"] = "/tmp/custom-validation.json"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {HELPER!s}; czechlynx_resolve_protocol; printf '%s\\n' \"${{CZECHLYNX_RESOLVED_VAL_INDEX}}\"",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "/tmp/custom-validation.json"


def test_invalid_protocol_fails():
    env = os.environ.copy()
    env["CZECHLYNX_SPLIT_PROTOCOL"] = "unknown"
    result = subprocess.run(
        ["bash", "-c", f"source {HELPER!s}; czechlynx_resolve_protocol"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "legacy" in result.stderr and "strict" in result.stderr


def test_both_czechlynx_trainers_use_the_shared_resolver():
    for name in ("train_czechlynx_rdd.sh", "train_czechlynx_loma.sh"):
        script = (ROOT / "slurm_scripts" / name).read_text()
        assert "czechlynx_protocol.sh" in script
        assert "czechlynx_resolve_protocol" in script
        assert "CZECHLYNX_RESOLVED_TRAIN_INDEX" in script
        assert "CZECHLYNX_RESOLVED_VAL_INDEX" in script
