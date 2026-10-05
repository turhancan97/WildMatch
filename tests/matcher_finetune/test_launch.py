"""`wildmatch finetune-matcher` plans what the lynx-finetuning wrappers ran.

Portable tests check the layout rules, the CzechLynx resolver against the original shell script,
that every planned flag is accepted by the trainer's own argument parser, and the safety rules.
The `data` tests compare plans with the paper's checkpoints on the cluster: the arguments recorded
in each LoMa checkpoint's metadata.json and the protocol JSON next to each checkpoint.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from wildmatch.matcher_finetune.launch import czechlynx_layout, plan_run

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "src" / "wildmatch" / "conf"
RESOLVER = ROOT / "slurm" / "matcher_finetune" / "czechlynx_protocol.sh"
# The shell resolver hard-codes the mining output root of the original checkout.
SHELL_MINING_OUTPUTS = Path("/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs")


def plan_for(overrides, tmp_path=None, profile="default"):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="finetune_matcher", overrides=[f"paths={profile}", *overrides])
    if tmp_path is not None:
        cfg.paths.data_root = str(tmp_path / "data")
        cfg.paths.checkpoint_root = str(tmp_path / "checkpoints")
        cfg.paths.external.mining_outputs = str(tmp_path / "mining")
        cfg.paths.external.loma_weights = str(tmp_path / "loma_B.pt")
        cfg.paths.external.rdd_weights_dir = str(tmp_path / "rdd-weights")
    return plan_run(OmegaConf.to_container(cfg, resolve=True))


def parse_with_trainer(plan, monkeypatch):
    if plan.matcher == "loma":
        from wildmatch.matcher_finetune.train_loma_matches import parse_args
    else:
        from wildmatch.matcher_finetune.train_by_lg_matches import parse_args
    monkeypatch.setattr(sys, "argv", ["trainer", *plan.trainer_args])
    return vars(parse_args())


@pytest.mark.parametrize("split", ["split-time_closed", "split-time_open"])
@pytest.mark.parametrize("protocol", ["legacy", "strict"])
@pytest.mark.parametrize("miner", ["loma", "rdd"])
def test_czechlynx_layout_matches_shell_resolver(split, protocol, miner):
    script = (
        f'source "{RESOLVER}"; czechlynx_resolve_protocol; '
        'printf "%s\\n" "$CZECHLYNX_RESOLVED_EXPERIMENT" "$CZECHLYNX_RESOLVED_TRAIN_INDEX" '
        '"$CZECHLYNX_RESOLVED_VAL_INDEX" "$CZECHLYNX_RESOLVED_OUTPUT_SUFFIX" "$CZECHLYNX_RESOLVED_SPLIT_SUFFIX"'
    )
    env = {"PATH": "/usr/bin:/bin", "CZECHLYNX_SPLIT_COLUMN": split, "CZECHLYNX_SPLIT_PROTOCOL": protocol,
           "CZECHLYNX_MINING_BACKEND": miner}  # fmt: skip
    shell = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True)
    lx = czechlynx_layout(split, protocol, miner, SHELL_MINING_OUTPUTS)
    expected = [lx["experiment"], str(lx["train_index"]), str(lx["val_index"]), lx["output_suffix"], lx["split_suffix"]]
    assert shell.stdout.splitlines() == expected


@pytest.mark.parametrize(
    "overrides",
    [
        ["dataset=salamander"],
        ["dataset=salamander", "matcher_finetune.component=descriptor"],
        ["dataset=salamander", "matcher_finetune=rdd"],
        ["dataset=salamander", "matcher_finetune=rdd", "matcher_finetune.component=descriptor"],
        ["dataset=czechlynx_closed"],
        ["dataset=czechlynx_closed", "matcher_finetune.component=joint"],
        ["dataset=czechlynx_open", "matcher_finetune=rdd", "matcher_finetune.component=joint"],
        ["dataset=czechlynx_closed", "matcher_finetune=rdd", "matcher_finetune.mined_by=rdd"],
    ],
)
def test_planned_flags_parse_with_the_trainer(overrides, tmp_path, monkeypatch):
    plan = plan_for([*overrides, "matcher_finetune.wandb.mode=disabled"], tmp_path)
    parsed = parse_with_trainer(plan, monkeypatch)
    assert Path(parsed["output_dir"]) == plan.output_dir
    assert Path(parsed["train_index"]) == plan.train_index
    json.loads(plan.protocol_text)
    assert plan.command[:2] == ["accelerate", "launch"]


def test_wildlife_layout(tmp_path):
    plan = plan_for(["dataset=salamander", "matcher_finetune=rdd", "matcher_finetune.mined_by=rdd"], tmp_path)
    base = tmp_path / "mining" / "wildlife-reid-10k" / "SalamanderID2025" / "indices" / "rdd"
    assert plan.train_index == base / "strong-matches_train_combined.json"
    assert plan.val_index == base / "strong-matches_test_combined.json"
    assert plan.data_root == tmp_path / "data" / "wildlife_processed" / "SalamanderID2025" / "legacy"
    assert plan.output_dir == tmp_path / "checkpoints/wildlife-reid-10k/SalamanderID2025/rdd-finetuned/legacy"
    assert plan.cache == tmp_path / "checkpoints/wildlife-reid-10k/SalamanderID2025/rdd-cache"
    assert plan.protocol_file.name == "wildlife_protocol.json"


def test_rdd_descriptor_defaults_to_one_image_and_eight_accumulation_steps(tmp_path, monkeypatch):
    plan = plan_for(["dataset=salamander", "matcher_finetune=rdd", "matcher_finetune.component=descriptor"], tmp_path)
    parsed = parse_with_trainer(plan, monkeypatch)
    assert (parsed["batch_size"], parsed["grad_accum_steps"]) == (1, 8)
    assert (parsed["trained_model"], parsed["rdd_train_component"]) == ("rdd", "descriptor")
    assert plan.cache is None  # only matcher-only RDD training reads the keypoint cache


def test_joint_is_czechlynx_only(tmp_path):
    with pytest.raises(ValueError, match="only defined for CzechLynx"):
        plan_for(["dataset=salamander", "matcher_finetune.component=joint"], tmp_path)


def test_existing_epochs_are_never_overwritten_and_auto_resumes_the_newest(tmp_path):
    overrides = ["dataset=czechlynx_closed"]
    output = plan_for(overrides, tmp_path).output_dir
    for epoch, complete in ((9, True), (19, True), (29, False)):
        (output / f"epoch_{epoch:03d}").mkdir(parents=True)
        if complete:
            (output / f"epoch_{epoch:03d}" / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="already holds epoch directories"):
        plan_for(overrides, tmp_path)
    plan = plan_for([*overrides, "matcher_finetune.resume=auto"], tmp_path)
    assert plan.resume == output / "epoch_019"
    assert plan.trainer_args[-2:] == ["--resume", str(output / "epoch_019")]


# ── against the paper's checkpoints (cluster only) ─────────────────────────────────────────────────

GMUM_CHECKPOINTS = Path("/shared/sets/datasets/vision/czechlynx/checkpoints")
# Registry key, matcher, checkpoint folder under the gmum checkpoint root, miner, (GPUs, batch per GPU).
PAPER_RUNS = [
    ("hyenaid2022", "loma", "wildlife-reid-10k/HyenaID2022/loma-finetuned/legacy-loma-mined", "loma", (4, 8)),
    ("leopardid2022", "loma", "wildlife-reid-10k/LeopardID2022/loma-finetuned/legacy-loma-mined", "loma", (4, 8)),
    ("seastarreid2023", "loma", "wildlife-reid-10k/SeaStarReID2023/loma-finetuned/legacy", "loma", (4, 8)),
    ("whaleshark", "loma", "wildlife-reid-10k/WhaleSharkID/loma-finetuned/legacy-rdd-mined", "rdd", (4, 8)),
    ("zindi", "loma", "wildlife-reid-10k/ZindiTurtleRecall/loma-finetuned/legacy-loma-mined", "loma", (4, 8)),
    ("salamander", "loma", "wildlife-reid-10k/SalamanderID2025/loma-finetuned/legacy-loma-mined", "loma", (2, 16)),
    ("czechlynx_closed", "loma", "czechlynx-time-closed/loma-b-finetuned-loma-mined-legacy", "loma", (4, 8)),
    ("czechlynx_open", "loma", "czechlynx-time-open/loma-b-finetuned-loma-mined-legacy", "loma", (4, 8)),
    ("nyala", "rdd", "wildlife-reid-10k/NyalaData/rdd-finetuned/legacy-rdd-mined-relaxed", "rdd", (4, 8)),
    ("salamander", "rdd", "wildlife-reid-10k/SalamanderID2025/rdd-finetuned/legacy-rdd-mined", "rdd", (4, 8)),
    ("czechlynx_closed", "rdd", "czechlynx-time-closed/rdd-finetuned-rdd-mined-legacy-relaxed", "rdd", (4, 8)),
]
# Fields that name the run rather than define it, or did not exist when the paper ran.
NOT_COMPARED = {"output_dir", "project", "run_name", "resume", "keep_every"}
# Index folders moved after a run: recorded prefix -> where the same files are today. WhaleSharkID's
# RDD-mined index lay in the shared indices/ folder until the per-miner subfolders existed; the files
# now in indices/rdd/ keep their mtime (2026-08-21, before the paper's run) and the old path is gone.
RELOCATED = {
    "/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/wildlife-reid-10k/WhaleSharkID/indices/": "/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/wildlife-reid-10k/WhaleSharkID/indices/rdd/",
}


def relocated(value):
    text = str(value)
    for old, new in RELOCATED.items():
        if text.startswith(old) and not Path(text).exists():
            return new + text[len(old) :]
    return text


def paper_plan(key, matcher, folder, miner, gpus_batch, tmp_path):
    gpus, batch = gpus_batch
    overrides = [f"dataset={key}", f"matcher_finetune={matcher}", f"matcher_finetune.mined_by={miner}",
                 f"matcher_finetune.num_processes={gpus}", f"matcher_finetune.batch_size={batch}",
                 f"matcher_finetune.output_dir={tmp_path / 'out'}"]  # fmt: skip
    return plan_for(overrides, profile="gmum")


@pytest.mark.data
@pytest.mark.parametrize("key, matcher, folder, miner, gpus_batch", [r for r in PAPER_RUNS if r[1] == "loma"])
def test_loma_plan_reproduces_the_recorded_arguments(key, matcher, folder, miner, gpus_batch, tmp_path, monkeypatch):
    metadata = GMUM_CHECKPOINTS / folder / "epoch_299" / "metadata.json"
    if not metadata.is_file():
        pytest.skip(f"{metadata} not available")
    recorded = json.loads(metadata.read_text())["args"]
    parsed = parse_with_trainer(paper_plan(key, matcher, folder, miner, gpus_batch, tmp_path), monkeypatch)
    differences = {
        name: (value, parsed.get(name))
        for name, value in recorded.items()
        if name not in NOT_COMPARED and str(parsed.get(name)) != relocated(value)
    }
    assert differences == {}


@pytest.mark.data
@pytest.mark.parametrize("key, matcher, folder, miner, gpus_batch", PAPER_RUNS)
def test_plan_reproduces_the_recorded_protocol(key, matcher, folder, miner, gpus_batch, tmp_path):
    directory = GMUM_CHECKPOINTS / folder
    files = [directory / n for n in ("czechlynx_protocol.json", "wildlife_protocol.json") if (directory / n).is_file()]
    if not files:
        pytest.skip(f"no protocol file in {directory}")
    recorded = json.loads(files[0].read_text())
    plan = paper_plan(key, matcher, folder, miner, gpus_batch, tmp_path)
    assert plan.protocol_file.name == files[0].name
    planned = json.loads(plan.protocol_text)
    differences = {
        name: (value, planned.get(name))
        for name, value in recorded.items()
        if str(planned.get(name)) != relocated(value)
    }
    assert differences == {}


def test_fewshot_training_reads_the_fewshot_view_and_indices(tmp_path, monkeypatch):
    plan = plan_for(["dataset=salamander", "matcher_finetune=rdd", "matcher_finetune.mined_by=rdd",
                     "matcher_finetune.fewshot.fraction=0.125"], tmp_path)  # fmt: skip
    root = tmp_path / "data" / "fewshot"
    view = "frac0.125-seed0"
    assert plan.data_root == root / "views/SalamanderID2025/legacy" / view
    assert (
        plan.train_index == root / "indices/SalamanderID2025/legacy" / view / "rdd/strong-matches_train_combined.json"
    )
    assert plan.output_dir == root / "checkpoints/SalamanderID2025/legacy" / view / "rdd-finetuned"
    assert plan.cache == tmp_path / "checkpoints/wildlife-reid-10k/SalamanderID2025/rdd-cache"
    parse_with_trainer(plan, monkeypatch)


def test_launch_provenance_is_appended_per_launch(tmp_path):
    from wildmatch.matcher_finetune.launch import PROVENANCE_FILE, record_launch

    overrides = ["dataset=salamander", "matcher_finetune=rdd", "matcher_finetune.seed=3"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="finetune_matcher", overrides=["paths=default", *overrides])
    cfg.paths.data_root = str(tmp_path / "data")
    cfg.paths.checkpoint_root = str(tmp_path / "checkpoints")
    cfg.paths.external.mining_outputs = str(tmp_path / "mining")
    cfg.paths.external.rdd_weights_dir = str(tmp_path / "rdd")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    plan = plan_run(cfg)
    plan.train_index.parent.mkdir(parents=True)
    plan.train_index.write_text("[]")
    record_launch(plan, cfg)
    record_launch(plan, cfg)
    data = json.loads((plan.output_dir / PROVENANCE_FILE).read_text())
    assert len(data["launches"]) == 2
    launch = data["launches"][0]
    assert launch["seed"] == 3 and launch["matcher"] == "rdd" and launch["command"] == plan.command
    assert launch["inputs"]["train_index"]["sha256"] == (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"  # sha256("[]")
    )
    assert launch["inputs"]["pretrained_rdd"]["exists"] is False
    assert launch["code"]["commit"]
    assert "diff" not in launch["code"]
