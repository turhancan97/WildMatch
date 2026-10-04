"""Paper checkpoints on the Hugging Face Hub: ``wildmatch weights list|verify|download|stage``.

``conf/weights.yaml`` lists every published file with its name in the Hub repository
(``hub``), its place under ``paths.checkpoint_root`` (``local``, the path the dataset
registry's ``checkpoints.custom.<matcher>`` names) and its SHA-256. ``download`` fetches
files into that place, so sweeps find them without extra settings; every file is checked
against its SHA-256 and a mismatch is deleted and reported. ``stage`` copies the local files
into a folder laid out like the Hub repository, for the maintainers to upload.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from omegaconf import OmegaConf

from wildmatch.paths import active_profile, load_paths

MATCHERS = ("loma", "rdd-lightglue")


class WeightsError(RuntimeError):
    """A checkpoint is missing, corrupt, or cannot be fetched."""


def load_manifest() -> Dict[str, Any]:
    source = resources.files("wildmatch") / "conf" / "weights.yaml"
    manifest = OmegaConf.to_container(OmegaConf.create(source.read_text(encoding="utf-8")), resolve=True)
    for entry in manifest["entries"]:
        if entry["matcher"] not in MATCHERS:
            raise WeightsError(f"unknown matcher in weights.yaml: {entry['matcher']}")
    return manifest


def select(entries: Iterable[Dict[str, Any]], datasets: Sequence[str] = (), matchers: Sequence[str] = ()) -> List[Dict[str, Any]]:
    entries = list(entries)
    known = {entry["dataset"] for entry in entries}
    unknown = set(datasets) - known
    if unknown:
        raise WeightsError(f"no published checkpoints for {', '.join(sorted(unknown))}; available: {', '.join(sorted(known))}")
    return [entry for entry in entries
            if (not datasets or entry["dataset"] in datasets) and (not matchers or entry["matcher"] in matchers)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_root(profile: Optional[str] = None) -> Path:
    return Path(str(load_paths(profile).checkpoint_root))


def file_status(root: Path, item: Dict[str, str]) -> str:
    path = root / item["local"]
    if not path.is_file():
        return "missing"
    return "ok" if sha256_file(path) == item["sha256"] else "mismatch"


def _hub_download(repo_id: str, filename: str, revision: str, token: Optional[str]) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token))


def download(entries: Iterable[Dict[str, Any]], root: Path, repo_id: str, revision: str = "main",
             token: Optional[str] = None, force: bool = False, log=print) -> List[Path]:
    """Fetch each entry's files into ``root``; files already present and valid are kept."""
    written = []
    for entry in entries:
        for item in entry["files"]:
            target = root / item["local"]
            if not force and target.is_file() and sha256_file(target) == item["sha256"]:
                log(f"ok        {item['local']}")
                continue
            source = _hub_download(repo_id, item["hub"], revision, token)
            if sha256_file(source) != item["sha256"]:
                raise WeightsError(f"{repo_id}:{item['hub']} does not match its recorded SHA-256; nothing written")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            os.close(fd)
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            log(f"fetched   {item['local']}")
            written.append(target)
    return written


def stage(entries: Iterable[Dict[str, Any]], root: Path, output: Path, log=print) -> Path:
    """Copy the local files into ``output`` with the Hub layout, after checking every SHA-256."""
    entries = list(entries)
    problems = [f"{item['local']}: {status}" for entry in entries for item in entry["files"]
                if (status := file_status(root, item)) != "ok"]
    if problems:
        raise WeightsError("cannot stage, local files are missing or changed:\n  " + "\n  ".join(problems))
    if output.exists() and any(output.iterdir()):
        raise WeightsError(f"staging folder is not empty: {output}")
    for entry in entries:
        for item in entry["files"]:
            target = output / item["hub"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / item["local"], target)
            log(f"staged    {item['hub']}")
    lines = ["| file | sha256 |", "| --- | --- |"]
    lines += [f"| `{item['hub']}` | `{item['sha256']}` |" for entry in entries for item in entry["files"]]
    (output / "SHA256SUMS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description="List, verify, download or stage the paper checkpoints.")
    sub = parser.add_subparsers(dest="action", required=True)
    for name, text in (("list", "show every published checkpoint and its local status"),
                       ("verify", "check local files against their SHA-256 (exit 1 on any problem)"),
                       ("download", "fetch checkpoints from the Hugging Face Hub"),
                       ("stage", "copy local files into a folder laid out like the Hub repository")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--dataset", action="append", default=[], help="registry key (repeatable; default all)")
        command.add_argument("--matcher", action="append", default=[], choices=MATCHERS)
        command.add_argument("--paths", help="path profile that sets checkpoint_root")
        if name == "download":
            command.add_argument("--repo", help="Hub repository (default: weights.yaml repo_id or WILDMATCH_HUB_REPO)")
            command.add_argument("--revision", help="Hub revision (default: weights.yaml revision)")
            command.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hub token for a private repository")
            command.add_argument("--force", action="store_true", help="download even when a valid local file exists")
        if name == "stage":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest()
        entries = select(manifest["entries"], args.dataset, args.matcher)
        root = checkpoint_root(active_profile(args.paths))
        if args.action in {"list", "verify"}:
            bad = 0
            for entry in entries:
                for item in entry["files"]:
                    status = file_status(root, item)
                    bad += status != "ok"
                    print(f"{status:<9} {entry['dataset']:<16} {entry['matcher']:<14} {item['local']}")
            print(f"checkpoint_root: {root}")
            return 1 if args.action == "verify" and bad else 0
        if args.action == "download":
            repo_id = args.repo or manifest.get("repo_id")
            if not repo_id:
                raise WeightsError("no Hub repository configured: pass --repo or set WILDMATCH_HUB_REPO")
            download(entries, root, repo_id, args.revision or manifest.get("revision") or "main", args.token, args.force)
            return 0
        stage(entries, root, args.output)
        print(f"Upload with: hf upload <repo> {args.output} . --repo-type model --private")
        return 0
    except WeightsError as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
