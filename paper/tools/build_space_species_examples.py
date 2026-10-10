#!/usr/bin/env python3
"""Build the Space's per-species pair examples from the datasets' original releases (CPU; run once).

The paper read these datasets through WildlifeReID-10k, whose terms forbid re-uploading its files, so the Space
never bundles them. The original releases carry their own licences, which allow redistribution with attribution
(user decision 2026-10-10): LeopardID2022, HyenaID2022 and WhaleSharkID (Wild Me on LILA BC) and SeaStarReID2023
(LILA BC) under CDLA-Permissive-1.0, ZindiTurtleRecall under CC BY-SA 4.0 (its competition rules); checked on the
source pages 2026-10-10. Nyala states no licence and is left out.
SalamanderID2025 reuses the two photos the project page already shows with the organisers' permission.

Steps:

    python paper/tools/build_space_species_examples.py select    # seeded random pair per species -> selection.csv
    python paper/tools/build_space_species_examples.py fetch --work <dir>   # stream the source archives
    python paper/tools/build_space_species_examples.py build --work <dir>   # crop, resize, write space/examples/species/

``select`` draws one identity on both sides of the split (seed 0, never by score), then one query (test) and one
database (train) photo of it, skipping whole-photo SAM 3 fallbacks. ``fetch`` extracts only the needed members
from each archive (tar streams, zip by HTTP range). ``build`` crops Wild Me photos with the source's COCO box (the
annotation of that photo and individual) and checks that the crop has the size of the
WildlifeReID-10k file; outputs are 640 px JPEGs, ``species.csv`` and ``ATTRIBUTION.md``. Refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "space" / "examples" / "species"
LONG_SIDE = 640
SEED = 0
LILA = "https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me"

# key -> registry entry, Space species label, archive, licence and credit of the original release.
SOURCES: Dict[str, dict] = {
    "leopard": dict(
        registry="leopardid2022",
        label="Leopard",
        url=f"{LILA}/leopard.coco.tar.gz",
        licence="CDLA-Permissive-1.0",
        credit="Leopard ID 2022: Botswana Predator Conservation Trust (2022), African Carnivore Wildbook, via Wild Me and LILA BC",
    ),
    "hyena": dict(
        registry="hyenaid2022",
        label="Spotted hyena",
        url=f"{LILA}/hyena.coco.tar.gz",
        licence="CDLA-Permissive-1.0",
        credit="Hyena ID 2022: Botswana Predator Conservation Trust (2022), African Carnivore Wildbook, via Wild Me and LILA BC",
    ),
    "whaleshark": dict(
        registry="whaleshark",
        label="Whale shark",
        url=f"{LILA}/whaleshark.coco.tar.gz",
        licence="CDLA-Permissive-1.0",
        credit="Whale Shark ID: Holmberg, Norman and Arzoumanian (2009), Endangered Species Research 7, via Wild Me and LILA BC",
    ),
    "seastar": dict(
        registry="seastarreid2023",
        label="Sea star",
        url="https://storage.googleapis.com/public-datasets-lila/sea-star-re-id/sea-star-re-id.zip",
        licence="CDLA-Permissive-1.0",
        credit="Sea Star Re-ID 2023: Wahltinez and Wahltinez (2024), Methods in Ecology and Evolution, via LILA BC",
    ),
    "turtle": dict(
        registry="zindi",
        label="Sea turtle",
        url="https://storage.googleapis.com/dm-turtle-recall/images.tar",
        licence="CC BY-SA 4.0",
        credit="Turtle Recall Conservation Challenge (Zindi, Local Ocean Conservation and Google DeepMind)",
    ),
}
# The SalamanderID2025 pair the project page shows (docs/assets/demo/before_after/), organisers' permission 2026-10-07.
# Pairs chosen by appearance instead (user decision 2026-10-10): the seed-0 draws were a head close-up against a
# whole body (leopard) and two close-ups of different parts of a star (sea star), so both photos share little visible
# pattern. Replacements were picked by eye from contact sheets of 8 seeded random individuals (seed 1), before any
# matcher ran: whole-body side views in two scenes (leopard), the whole star from above in two tubs (sea star).
BY_APPEARANCE = {
    "leopard": (
        "LeopardID2022/leopard.coco/images/train2022/000000006296_6310.jpg",
        "LeopardID2022/leopard.coco/images/train2022/000000006126_6140.jpg",
    ),
    "seastar": (
        "SeaStarReID2023/sea-star-re-id/Anau14/IMG_3470_088c707598221cc9.jpg",
        "SeaStarReID2023/sea-star-re-id/Anau14/IMG_3466_bc71c73035426111.jpg",
    ),
}
SALAMANDER = dict(
    label="Fire salamander",
    query="query/images/616e46c4c91594a7_795.jpg",
    gallery="database/images/25372f49baf1e44c_305.jpg",
    licence="AnimalCLEF2025 rules; shown with the organisers' permission (non-commercial)",
    credit="SalamanderID2025 (AnimalCLEF 2025)",
)


def select(work: Path) -> pd.DataFrame:
    from wildmatch.data.registry import load_dataset

    rows = []
    for key, source in SOURCES.items():
        entry = load_dataset(source["registry"])
        data = pd.read_csv(Path(entry.root) / entry.metadata_file)
        data = data[~data["sam3_full_frame"].astype(bool)]
        rng = np.random.default_rng(SEED)
        test, train = data[data["split"] == "test"], data[data["split"] == "train"]
        ids = sorted(set(test["identity"]) & set(train["identity"]))
        identity = ids[rng.integers(len(ids))]
        query = test[test["identity"] == identity]
        gallery = train[train["identity"] == identity]
        picked = [query.iloc[rng.integers(len(query))], gallery.iloc[rng.integers(len(gallery))]]
        chosen_by = "seeded random (seed 0)"
        if key in BY_APPEARANCE:
            picked = [data[data["original_path"] == path].iloc[0] for path in BY_APPEARANCE[key]]
            if [r["split"] for r in picked] != ["test", "train"] or picked[0]["identity"] != picked[1]["identity"]:
                raise SystemExit(f"{key}: the pair chosen by appearance is not one individual's test/train photos")
            identity, chosen_by = picked[0]["identity"], "by appearance"
        for side, row in zip(("query", "gallery"), picked):
            height, width = json.loads(row["mask"])["size"]
            rows.append(
                dict(
                    species=key,
                    side=side,
                    identity=identity,
                    wr10k_path=row["original_path"],
                    chosen=chosen_by,
                    width=width,
                    height=height,
                )
            )
    table = pd.DataFrame(rows)
    work.mkdir(parents=True, exist_ok=True)
    table.to_csv(work / "selection.csv", index=False)
    return table


def source_member(row: pd.Series) -> str:
    """Archive member name of the original photo behind a WildlifeReID-10k path."""
    name = Path(row["wr10k_path"]).name
    if row["species"] in ("leopard", "hyena", "whaleshark"):
        return name.rsplit("_", 1)[0] + ".jpg"  # <image id>_<annotation id>.jpg
    if row["species"] == "seastar":
        return f"{Path(row['wr10k_path']).parent.name}/{name.rsplit('_', 1)[0]}.jpg"  # <photo>_<hash>.jpg
    return name.split("_ID_")[0] + Path(name).suffix  # turtle: <id>_<id>.JPG


class HttpFile:
    """Seekable read-only view of a URL through HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(request).read()
        self.pos += len(data)
        return data


def fetch(work: Path) -> None:
    table = pd.read_csv(work / "selection.csv")
    for key, source in SOURCES.items():
        target = work / key
        target.mkdir(parents=True, exist_ok=True)
        members = [source_member(row) for _, row in table[table["species"] == key].iterrows()]
        if all(any(target.rglob(Path(m).name)) for m in members):
            continue
        if source["url"].endswith(".zip"):
            archive = zipfile.ZipFile(HttpFile(source["url"]))
            for name in archive.namelist():
                if any(name.endswith("/" + m) for m in members):
                    (target / Path(name).name).write_bytes(archive.read(name))
            continue
        patterns = [f"*/{m}" for m in members] + (["*/annotations/*"] if key != "turtle" else [])
        flags = "-xz" if source["url"].endswith(".gz") else "-x"
        curl = subprocess.Popen(["curl", "-sf", "--retry", "3", source["url"]], stdout=subprocess.PIPE)
        subprocess.run(
            ["tar", flags, "-C", str(target), "--wildcards", "--warning=no-unknown-keyword", *patterns],
            stdin=curl.stdout,
            check=True,
        )
        if curl.wait() != 0:
            raise SystemExit(f"{key}: download failed")


def coco_boxes(folder: Path) -> Dict[tuple, list]:
    """(image file name, individual) -> box. The ``name`` field is the individual's id, as in WildlifeReID-10k."""
    boxes: Dict[tuple, list] = {}
    for path in (p for p in folder.rglob("annotations/*.json") if not p.name.startswith("._")):
        data = json.loads(path.read_text())
        files = {image["id"]: image["file_name"] for image in data["images"]}
        for annotation in data["annotations"]:
            boxes.setdefault((files[annotation["image_id"]], annotation["name"]), []).append(annotation["bbox"])
    return boxes


def thumbnail(image: Image.Image) -> Image.Image:
    scale = LONG_SIDE / max(image.size)
    if scale >= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def original_photo(work: Path, row: pd.Series, boxes: Dict[tuple, list]) -> Image.Image:
    member = Path(source_member(row)).name
    matches = sorted((work / row["species"]).rglob(member))
    if len(matches) != 1:
        raise SystemExit(f"{row['species']}: {member} found {len(matches)} times under {work}")
    image = Image.open(matches[0]).convert("RGB")
    if row["species"] in ("leopard", "hyena", "whaleshark"):
        found = boxes.get((member, row["identity"]), [])
        if len(found) != 1:
            raise SystemExit(f"{row['species']}: {len(found)} boxes for {member} and {row['identity']}")
        x, y, w, h = found[0]
        # Same rounding as the WildlifeReID-10k crops (checked by size below); a box may end one pixel past the photo.
        image = image.crop((round(x), round(y), min(round(x + w), image.width), min(round(y + h), image.height)))
    if abs(image.width - row["width"]) > 1 or abs(image.height - row["height"]) > 1:
        raise SystemExit(
            f"{row['species']}: {member} gives {image.size}, WildlifeReID-10k has {(row['width'], row['height'])}"
        )
    return image


def build(work: Path, overwrite: bool) -> List[dict]:
    from wildmatch.data.registry import load_dataset

    if OUT.exists() and any(OUT.iterdir()) and not overwrite:
        raise SystemExit(f"{OUT} exists; pass --overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(work / "selection.csv")
    rows = []
    for key, source in SOURCES.items():
        boxes = coco_boxes(work / key) if key in ("leopard", "hyena", "whaleshark") else {}
        for _, row in table[table["species"] == key].iterrows():
            name = f"{key}_{row['side']}.jpg"
            thumbnail(original_photo(work, row, boxes)).save(OUT / name, quality=92)
            rows.append(
                dict(
                    file=name,
                    species=source["label"],
                    side=row["side"],
                    identity=row["identity"],
                    source=f"{source['credit']}: {source_member(row)}",
                    chosen=row["chosen"],
                    licence=source["licence"],
                )
            )
    root = Path(load_dataset("salamander").root)
    for side in ("query", "gallery"):
        name = f"salamander_{side}.jpg"
        thumbnail(Image.open(root / SALAMANDER[side]).convert("RGB")).save(OUT / name, quality=92)
        rows.append(
            dict(
                file=name,
                species=SALAMANDER["label"],
                side=side,
                identity="",
                source=f"{SALAMANDER['credit']}: {SALAMANDER[side]}",
                chosen="the project page's pair",
                licence=SALAMANDER["licence"],
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "species.csv", index=False)
    credits = "\n".join(f"- {s['label']}: {s['credit']}, {s['licence']}." for s in [*SOURCES.values(), SALAMANDER])
    (OUT / "ATTRIBUTION.md").write_text(
        "# Per-species example pairs\n\n"
        "One individual per species, a query (test split) and a gallery (training split) photo, drawn by seeded random "
        f"sampling (seed {SEED}) from the paper's splits, except Leopard and Sea star, chosen by appearance (both photos "
        "show the same view; see `chosen` in species.csv), never by score, and taken from each dataset's original release "
        "(Wild Me photos cropped with the release's own box). The fire salamander pair is the one the project page shows.\n\n"
        f"{credits}\n\nModified: cropped and resized to {LONG_SIDE} px. The sea turtle photos stay under CC BY-SA 4.0.\n",
        encoding="utf-8",
    )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["select", "fetch", "build"])
    parser.add_argument("--work", type=Path, default=Path("species-sources"), help="download folder (not committed)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.step == "select":
        print(select(args.work).to_string())
    elif args.step == "fetch":
        fetch(args.work)
    else:
        print(pd.DataFrame(build(args.work, args.overwrite)).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
