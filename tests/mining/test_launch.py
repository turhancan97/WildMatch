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
