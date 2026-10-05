"""The few-shot view must keep the budget, the positives guarantee and the cache keys."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wildmatch.mining.wildlife_dataset import WildlifeRecord  # noqa: E402
from wildmatch.mining.wildlife_fewshot import (  # noqa: E402
    UNUSED_SPLIT,
    allocate_budget,
    build_subset,
    fraction_tag,
    prepare_fewshot,
    select_frames,
    split_column,
    view_name,
)


def test_fraction_tags_are_stable():
    assert [fraction_tag(f) for f in (0.125, 0.25, 0.5, 1.0)] == ["0.125", "0.25", "0.5", "1.0"]
    assert view_name(0.125, 0) == "frac0.125-seed0"
    assert split_column(0.5, 3) == "split_frac0.5_seed3"


def test_budget_is_exact_and_proportional_among_free_identities():
    counts = {"a": 40, "b": 16, "c": 8, "d": 5}
    for fraction in (0.125, 0.25, 0.5):
        kept = allocate_budget(counts, fraction)
        assert sum(kept.values()) == round(fraction * sum(counts.values()))
        assert all(min(n, 2) <= kept[name] <= n for name, n in counts.items())
        # identities above the minimum share one scale (proportional up to rounding)
        free = {name: kept[name] / n for name, n in counts.items() if kept[name] > 2}
        if len(free) > 1:
            scales = sorted(free.values())
            assert scales[-1] - scales[0] <= 1 / min(counts[name] for name in free)


def test_lifted_identities_are_compensated_by_the_rich_ones():
    # f=0.125 of 69 frames = 9; every identity keeps >= 2, so 'a' gets 3 instead of 5
    kept = allocate_budget({"a": 40, "b": 16, "c": 8, "d": 5}, 0.125)
    assert kept == {"a": 3, "b": 2, "c": 2, "d": 2}


def test_no_rounding_drift_at_full_and_half_fraction():
    counts = {f"id{i}": n for i, n in enumerate([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])}
    assert allocate_budget(counts, 1.0) == counts
    half = allocate_budget(counts, 0.5)
    assert sum(half.values()) == round(0.5 * sum(counts.values()))
    assert half["id0"] == 1 and half["id1"] == 2 and half["id2"] == 2


def test_full_fraction_keeps_everything():
    counts = {"a": 3, "b": 1, "c": 7}
    assert allocate_budget(counts, 1.0) == counts


def test_minimum_of_two_frames_per_identity_is_never_violated():
    counts = {f"id{i}": 2 for i in range(10)}
    kept = allocate_budget(counts, 0.125)
    assert all(v == 2 for v in kept.values())  # infeasible budget -> minimums kept
    assert sum(kept.values()) > round(0.125 * 20)


def test_pre_existing_singletons_stay_single():
    counts = {"solo": 1, "rich": 80, "other": 40}
    kept = allocate_budget(counts, 0.25)
    assert kept["solo"] == 1
    assert sum(kept.values()) == round(0.25 * 121)


def test_extra_frames_go_by_largest_remainder():
    # budget = round(0.25 * 29) = 7; 'lifted' (1.75) is pinned at 2, the remaining 5 frames
    # are shared by 10:12 -> 2.27 / 2.73 -> floors 2 / 2, the extra frame goes to 'exact'
    kept = allocate_budget({"lifted": 7, "half": 10, "exact": 12}, 0.25)
    assert kept == {"lifted": 2, "half": 2, "exact": 3}


def test_selection_is_deterministic_and_nested():
    frames = [f"train/x/x/frame_{i:06d}.jpg" for i in range(20)]
    small = select_frames(frames, 3, seed=0, identity="x")
    large = select_frames(frames, 9, seed=0, identity="x")
    assert set(small) <= set(large)
    assert select_frames(frames, 9, seed=0, identity="x") == large
    assert select_frames(frames, 9, seed=1, identity="x") != large


def _record(split: str, identity: str, index: int) -> WildlifeRecord:
    return WildlifeRecord(
        dataset_id="Toy",
        identity=identity,
        collection=identity,
        original_path=f"masked_images/Toy/{split}/{identity}_{index}.jpg",
        canonical_path=f"{split}/{identity}/{identity}/frame_{index:06d}.jpg",
        official_split=split,
        generated_split=split,
    )


def _toy_records() -> list[WildlifeRecord]:
    records = []
    for identity, n in (("a", 16), ("b", 8), ("c", 4), ("d", 2), ("e", 1)):
        records += [_record("train", identity, i) for i in range(n)]
    for identity in ("a", "b", "c", "d", "e"):
        records += [_record("test", identity, i) for i in range(2)]
    return records


def test_build_subset_keeps_test_and_guarantees_positives():
    records = _toy_records()
    subset, summary = build_subset(records, 0.25, seed=0, min_per_identity=2)
    train = [r for r in subset if r.generated_split == "train"]
    test = [r for r in subset if r.generated_split == "test"]
    assert len(test) == 10
    assert summary["train_frames_full"] == 31
    assert summary["train_frames_kept"] == len(train)
    per_identity = summary["per_identity"]
    assert per_identity["e"]["kept"] == 1 and summary["singleton_identities"] == 1
    for identity in ("a", "b", "c", "d"):
        assert per_identity[identity]["kept"] >= 2
    assert summary["identities_lifted_to_minimum"] == 2  # c: 0.25 * 4 = 1, d: 0.25 * 2 = 0.5
    # 2+2+2+2+1 minimum frames exceed the budget of round(0.25 * 31) = 8 -> reported as infeasible
    assert summary["train_frames_kept"] == 9 and not summary["budget_feasible"]
    assert summary["effective_fraction"] == pytest.approx(9 / 31)

    _, half = build_subset(records, 0.5, seed=0, min_per_identity=2)
    assert half["train_frames_kept"] == round(0.5 * 31) and half["budget_feasible"]
    assert half["per_identity"]["d"]["kept"] == 2 and half["per_identity"]["e"]["kept"] == 1


def _write_full_view(root: Path, records: list[WildlifeRecord]) -> Path:
    images = root / "images"
    view = root / "full"
    metadata = root / "metadata.csv"
    rows = []
    for record in records:
        image = images / record.original_path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"jpg")
        link = view / record.canonical_path
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(image)
        rows.append({"identity": record.identity, "path": record.original_path, "split": record.official_split})
    rows.append({"identity": "unknown", "path": "masked_images/Toy/train/unknown_0.jpg", "split": "train"})
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["identity", "path", "split"])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "dataset_id": "Toy",
        "protocol": "legacy",
        "config": {"dataset_id": "Toy", "metadata_csv": str(metadata)},
        "records": [record.__dict__ for record in records],
    }
    (view / "manifest.json").write_text(json.dumps(manifest))
    return view


def test_prepare_fewshot_writes_nested_views_and_metadata(tmp_path: Path):
    records = _toy_records()
    view = _write_full_view(tmp_path, records)
    metadata_out = tmp_path / "metadata_fewshot.csv"
    outputs = {}
    for fraction in (0.125, 0.5, 1.0):
        out = tmp_path / "views" / view_name(fraction, 0)
        summary = prepare_fewshot(view, out, fraction, seed=0, metadata_out=metadata_out)
        outputs[fraction] = out
        # canonical names and link targets are those of the full view
        for link in out.rglob("frame_*.jpg"):
            rel = link.relative_to(out)
            assert link.is_symlink()
            assert (view / rel).is_symlink()
            assert link.resolve() == (view / rel).resolve()
        assert sorted(p.relative_to(out) for p in (out / "test").rglob("frame_*.jpg")) == sorted(
            p.relative_to(view) for p in (view / "test").rglob("frame_*.jpg")
        )
        kept_train = sorted((out / "train").rglob("frame_*.jpg"))
        assert len(kept_train) == summary["fewshot"]["train_frames_kept"]
        assert json.loads((out / "fewshot.json").read_text())["view"] == view_name(fraction, 0)
    small = {p.relative_to(outputs[0.125]) for p in (outputs[0.125] / "train").rglob("frame_*.jpg")}
    half = {p.relative_to(outputs[0.5]) for p in (outputs[0.5] / "train").rglob("frame_*.jpg")}
    full = {p.relative_to(outputs[1.0]) for p in (outputs[1.0] / "train").rglob("frame_*.jpg")}
    assert small < half < full
    assert len(full) == 31

    with metadata_out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = {split_column(f, 0) for f in (0.125, 0.5, 1.0)}
    assert columns <= set(rows[0])
    for row in rows:
        for column in columns:
            if row["split"] == "test":
                assert row[column] == "test"
            elif row["identity"] == "unknown":
                assert row[column] == UNUSED_SPLIT  # excluded by the canonical view
            else:
                assert row[column] in {"train", UNUSED_SPLIT}
    full_column = split_column(1.0, 0)
    assert sum(1 for row in rows if row[full_column] == "train") == 31
    small_column = split_column(0.125, 0)
    assert sum(1 for row in rows if row[small_column] == "train") == len(small)


def test_prepare_fewshot_refuses_other_parameters_without_force(tmp_path: Path):
    records = _toy_records()
    view = _write_full_view(tmp_path, records)
    out = tmp_path / "views" / "x"
    first = prepare_fewshot(view, out, 0.5, seed=0, metadata_out=None)
    assert "reused_existing_view" not in first["fewshot"]
    again = prepare_fewshot(view, out, 0.5, seed=0, metadata_out=None)  # re-running is idempotent
    assert again["fewshot"]["reused_existing_view"] is True  # complete view: links not rewritten
    # an interrupted run (no records.jsonl yet) is finished by re-verifying the links
    (out / "records.jsonl").unlink()
    resumed = prepare_fewshot(view, out, 0.5, seed=0, metadata_out=None)
    assert "reused_existing_view" not in resumed["fewshot"]
    assert (out / "records.jsonl").is_file()
    with pytest.raises(FileExistsError):
        prepare_fewshot(view, out, 0.5, seed=1, metadata_out=None)
    summary = prepare_fewshot(view, out, 0.5, seed=1, metadata_out=None, force=True)
    kept = {str(p.relative_to(out)) for p in (out / "train").rglob("frame_*.jpg")}
    manifest = json.loads((out / "manifest.json").read_text())
    expected = {r["canonical_path"] for r in manifest["records"] if r["generated_split"] == "train"}
    assert kept == expected  # links of the seed-0 selection were pruned
    assert len(kept) == summary["fewshot"]["train_frames_kept"]


def test_concurrent_metadata_writers_keep_every_column(tmp_path: Path):
    import multiprocessing

    records = _toy_records()
    view = _write_full_view(tmp_path, records)
    metadata_out = tmp_path / "metadata_fewshot.csv"
    fractions = (0.125, 0.25, 0.5, 1.0)
    procs = [
        multiprocessing.Process(
            target=prepare_fewshot,
            args=(view, tmp_path / "views" / view_name(f, 0), f),
            kwargs={"seed": 0, "metadata_out": metadata_out},
        )
        for f in fractions
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert all(p.exitcode == 0 for p in procs)
    with metadata_out.open(newline="") as handle:
        header = next(csv.reader(handle))
    assert {split_column(f, 0) for f in fractions} <= set(header)
    assert not (metadata_out.with_name(metadata_out.name + ".lock")).exists()
