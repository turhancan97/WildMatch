#!/usr/bin/env python
"""Mux the narration onto the rendered Manim video, write captions, and place the results in
``out/`` (gitignored). Run after ``manim`` in the wm-video environment::

    manim -qh --fps 30 --media_dir media -o explainer.mp4 explainer.py Explainer
    python build.py

Outputs: ``out/wildmatch_explainer.mp4`` (H.264 + AAC), ``out/wildmatch_explainer.srt`` and
``out/transcript.md``. The MP4 is hosted unlisted and never committed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_render(media: Path) -> Path:
    candidates = sorted(media.glob("videos/explainer/*/explainer.mp4"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no Manim render under {media}")
    return candidates[-1]


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_caption(text: str, max_chars: int = 92) -> list[str]:
    """Split a shot's narration into caption lines at sentence ends, then at commas."""
    parts = re.split(r"(?<=[.:;])\s+", text.strip())
    out: list[str] = []
    for part in parts:
        while len(part) > max_chars:
            cut = part.rfind(", ", 0, max_chars)
            if cut <= 0:
                cut = part.rfind(" ", 0, max_chars)
            out.append(part[:cut + 1].strip())
            part = part[cut + 1:].strip()
        out.append(part)
    return [p for p in out if p]


def write_captions(timings: dict, out_dir: Path) -> Path:
    lines = []
    index = 1
    for shot in timings["shots"]:
        pieces = split_caption(shot["text"])
        words = [len(p.split()) for p in pieces]
        total = sum(words) or 1
        t = float(shot["start"])
        for piece, w in zip(pieces, words):
            dur = float(shot["duration"]) * w / total
            lines.append(f"{index}\n{srt_time(t)} --> {srt_time(t + dur)}\n{piece}\n")
            index += 1
            t += dur
    path = out_dir / "wildmatch_explainer.srt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    transcript = out_dir / "transcript.md"
    transcript.write_text("\n\n".join(s["text"] for s in timings["shots"]) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--media", type=Path, default=HERE / "media")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()
    timings = json.loads((HERE / "audio" / "timings.json").read_text(encoding="utf-8"))
    render = find_render(args.media)
    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / "wildmatch_explainer.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(render), "-i", str(HERE / "audio" / "narration.wav"),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(final),
    ], check=True)
    srt = write_captions(timings, args.out)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(final)],
                           capture_output=True, text=True, check=True)
    print(f"[explainer] {final} ({float(probe.stdout):.1f} s, {final.stat().st_size // 1024} KB); captions {srt}")


if __name__ == "__main__":
    main()
