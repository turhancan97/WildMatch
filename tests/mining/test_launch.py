"""`wildmatch mine` plans what the rdd-parallel-benchmark wrappers ran.

Portable tests use the default path profile pointed at a temporary tree; the `data` test compares the
planned task with the arguments the paper's SalamanderID2025 mining tasks recorded in their files.
"""

import json
from pathlib import Path

import pytest

from wildmatch.mining.launch import main, plan_mining


@pytest.fixture
def tmp_profile(tmp_path, monkeypatch):
    for name, value in {
        "WILDMATCH_DATA_ROOT": tmp_path / "data",
        "WILDMATCH_CHECKPOINT_ROOT": tmp_path / "checkpoints",
        "WILDMATCH_MINING_OUTPUTS": tmp_path / "mining",
        "WILDMATCH_LOMA_WEIGHTS": tmp_path / "loma_B.pt",
        "WILDMATCH_RDD_WEIGHTS_DIR": tmp_path / "rdd",
    }.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.delenv("WILDMATCH_PATHS", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_wildlife_layout(tmp_profile):
    plan = plan_mining("salamander", "loma", "legacy", "default")
    assert plan.view == tmp_profile / "data/wildlife_processed/SalamanderID2025/legacy"
    assert plan.cache == tmp_profile / "checkpoints/wildlife-reid-10k/SalamanderID2025/loma-cache"
    assert plan.report == tmp_profile / "mining/wildlife-reid-10k/SalamanderID2025/indices/loma/strong-matches"
    assert plan.splits == ("train", "test")
    assert plan_mining("salamander", "rdd", "strict", "default").splits == ("train", "val", "test")
    task = plan.task_command("train", 7)
    assert task[1:3] == ["-m", "wildmatch.mining.wildlife_mine"]
    assert task[task.index("--lg_weights") + 1] == str(tmp_profile / "loma_B.pt")


def test_czechlynx_layout(tmp_profile):
    plan = plan_mining("czechlynx_open", "rdd", "legacy", "default")
    assert plan.view == tmp_profile / "data/CzechLynx_processed_time_open"
    assert plan.cache == tmp_profile / "checkpoints/czechlynx-time-open/rdd-cache"
    assert plan.report == tmp_profile / "mining/czechlynx-time-open/legacy/rdd/strong-matches"
    assert plan.splits == ("train", "val", "test")
    task = plan.task_command("val", 3)
    assert task[task.index("--rdd_weights") + 1] == str(tmp_profile / "rdd/RDD-v2.pth")
    assert task[task.index("--lg_weights") + 1] == str(tmp_profile / "rdd/RDD_lg-v2.pth")
    assert plan_mining("czechlynx_closed", "loma", "legacy", "default").cache.name == "loma-b-cache"


def test_task_never_overwrites_a_mined_query(tmp_profile, capsys):
    plan = plan_mining("salamander", "loma", "legacy", "default")
    output = Path(f"{plan.report}_train_4.json")
    output.parent.mkdir(parents=True)
    output.write_text("{}")
    args = ["task", "--dataset", "salamander", "--backend", "loma", "--paths", "default", "--split", "train"]
    assert main([*args, "--index", "4"]) == 1
    assert "refusing" in capsys.readouterr().err
    assert output.read_text() == "{}"


def test_submit_refuses_an_existing_index(tmp_profile, capsys):
    plan = plan_mining("salamander", "rdd", "legacy", "default")
    Path(f"{plan.report}_train_combined.json").parent.mkdir(parents=True)
    Path(f"{plan.report}_train_combined.json").write_text("[]")
    assert main(["submit", "--dataset", "salamander", "--backend", "rdd", "--paths", "default", "--dry-run"]) == 1
    assert "refusing" in capsys.readouterr().err


def test_report_option_redirects_tasks_and_aggregation(tmp_profile, capsys):
    target = tmp_profile / "elsewhere" / "strong-matches"
    assert main(["plan", "--dataset", "salamander", "--backend", "loma", "--paths", "default",
                 "--report", str(target)]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert f"report={target}" in out and f"--dump_report {target}" in out


PAPER_QUERY = Path(
    "/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/wildlife-reid-10k/SalamanderID2025/indices"
)


@pytest.mark.data
@pytest.mark.parametrize("backend", ["loma", "rdd"])
def test_task_reproduces_the_paper_mining_arguments(backend):
    recorded_file = PAPER_QUERY / backend / "strong-matches_train_0.json"
    if not recorded_file.is_file():
        pytest.skip(f"{recorded_file} not available")
    recorded = json.loads(recorded_file.read_text())
    plan = plan_mining("salamander", backend, "legacy", "gmum")
    task = plan.task_command("train", 0)
    value = lambda flag: task[task.index(flag) + 1]  # noqa: E731
    assert value("--dataset_id") == recorded["dataset"]
    assert value("--cache_dir") == recorded["cache_dir"]
    assert value("--lg_weights") == recorded["weights"]
    if backend == "loma":  # the RDD miner records no variant (the wrapper passed loma-b anyway)
        assert value("--variant") == recorded["variant"]
    assert int(value("--frames_per_collection")) == recorded["frames_per_collection"]
    assert int(value("--top_k_frames")) == recorded["top_k_frames"]
    assert int(value("--top_m")) == recorded["top_m"]
    assert recorded["selected_frames"][0]["query_frame"].startswith(str(plan.view) + "/")
    assert Path(value("--dump_report")).parent == recorded_file.parent


def test_fewshot_plan_shares_the_full_view_caches(tmp_profile):
    full = plan_mining("salamander", "loma", "legacy", "default")
    few = plan_mining("salamander", "loma", "legacy", "default", fraction=0.25, seed=1)
    root = tmp_profile / "data" / "fewshot"
    assert few.view == root / "views/SalamanderID2025/legacy/frac0.25-seed1"
    assert few.report == root / "indices/SalamanderID2025/legacy/frac0.25-seed1/loma/strong-matches"
    assert few.cache == full.cache
    assert few.cache_command[few.cache_command.index("--dataset_root") + 1] == str(full.view)
    assert few.view_command[2] == "wildmatch.mining.wildlife_fewshot"
    assert few.view_command[few.view_command.index("--source_view") + 1] == str(full.view)
    # the probe metadata copy goes to the few-shot root, never into the dataset folder
    assert few.view_command[few.view_command.index("--metadata_out") + 1].startswith(str(root))


def test_fewshot_czechlynx_is_time_closed_only(tmp_profile):
    plan = plan_mining("czechlynx_closed", "rdd", "legacy", "default", fraction=0.5)
    assert plan.view_command[2] == "wildmatch.mining.czechlynx_fewshot"
    with pytest.raises(SystemExit, match="split-time_closed only"):
        plan_mining("czechlynx_open", "rdd", "legacy", "default", fraction=0.5)


def _fake_inputs(tmp_profile, plan):
    """A two-identity view, a LoMa cache covering it, and weights: enough for check and submit."""
    import numpy as np

    for split in plan.splits:
        for identity in ("a", "b"):
            frame = plan.view / split / identity / "c0" / "frame_000000.jpg"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"jpg")
            npz = plan.cache / frame.relative_to(plan.view).with_suffix(".npz")
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz, keypoints=np.zeros((1, 2)), descriptors=np.zeros((1, 4)), scores=np.zeros(1),
                     image_size=np.array([4, 4]))  # fmt: skip
    (plan.cache / "manifest.json").write_text(json.dumps(
        {"backend": "loma", "variant": "loma-b", "resize": 512, "num_keypoints": 512, "patch_size": 14}))  # fmt: skip
    plan.weights.write_bytes(b"weights")


def test_submit_freezes_the_run_and_tasks_execute_only_the_frozen_command(tmp_profile, monkeypatch, capsys):
    import wildmatch.mining.launch as launch

    plan = plan_mining("salamander", "loma", "legacy", "default")
    _fake_inputs(tmp_profile, plan)
    assert main(["submit", "--dataset", "salamander", "--backend", "loma", "--paths", "default", "--dry-run"]) == 0
    submission = next((tmp_profile / "logs/mining/submissions").glob("*/submission.json"))
    record = json.loads(submission.read_text())
    assert record["counts"] == {"train": 2, "test": 2}
    assert record["inputs"]["weights"]["sha256"] and record["code"]["commit"]
    assert f"task --submission {submission} --split train" in capsys.readouterr().out

    calls = []
    monkeypatch.setattr(launch.subprocess, "call", lambda command: calls.append(command) or 0)
    # The registry/profile may change after submission; the task still runs the frozen command.
    monkeypatch.setenv("WILDMATCH_MINING_OUTPUTS", str(tmp_profile / "moved"))
    assert main(["task", "--submission", str(submission), "--split", "test", "--index", "1"]) == 0
    assert calls[-1][calls[-1].index("--query_id") + 1] == "1"
    assert calls[-1][calls[-1].index("--dump_report") + 1] == str(plan.report)


def test_frozen_task_fails_closed_when_an_input_changed(tmp_profile, monkeypatch, capsys):
    import wildmatch.mining.launch as launch

    plan = plan_mining("salamander", "loma", "legacy", "default")
    _fake_inputs(tmp_profile, plan)
    main(["submit", "--dataset", "salamander", "--backend", "loma", "--paths", "default", "--dry-run"])
    submission = next((tmp_profile / "logs/mining/submissions").glob("*/submission.json"))
    monkeypatch.setattr(launch.subprocess, "call", lambda command: pytest.fail("must not run"))
    plan.weights.write_bytes(b"other weights")
    assert main(["task", "--submission", str(submission), "--split", "train", "--index", "0"]) == 1
    assert "input weights changed since submission" in capsys.readouterr().err


def test_frozen_task_rejects_an_index_outside_the_split(tmp_profile, monkeypatch):
    import wildmatch.mining.launch as launch

    plan = plan_mining("salamander", "loma", "legacy", "default")
    _fake_inputs(tmp_profile, plan)
    main(["submit", "--dataset", "salamander", "--backend", "loma", "--paths", "default", "--dry-run"])
    submission = next((tmp_profile / "logs/mining/submissions").glob("*/submission.json"))
    monkeypatch.setattr(launch.subprocess, "call", lambda command: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="outside the 2 train collections"):
        main(["task", "--submission", str(submission), "--split", "train", "--index", "2"])
