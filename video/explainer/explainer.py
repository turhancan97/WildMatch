"""WildMatch home-page explainer (Manim Community).

Seven shots cut to the Kokoro narration in ``audio/timings.json``; the script and shot
list are in ``script.md``. Photos, correspondences and mining pools are the committed demo
exports (``docs/assets/demo/before_after`` and ``docs/assets/demo/mined_pairs``), so
nothing on screen is drawn by hand. Brand palette on white. No venue, status or author
names appear.

Render (wm-video environment, from this directory)::

    manim -qh --fps 30 --media_dir media -o explainer.mp4 explainer.py Explainer
    python build.py   # muxes the narration, writes captions, copies to out/

"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, UL, ORIGIN, PI, WHITE, BLACK,
    AnimationGroup, Arrow, Brace, Create, Dot, FadeIn, FadeOut, Group, ImageMobject, LaggedStart,
    Line, NumberLine, Rectangle, RoundedRectangle, Scene, Text, Transform, VGroup, Write, config,
    rate_functions, SurroundingRectangle,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEMO = REPO / "docs" / "assets" / "demo"
LOGO = REPO / "docs" / "assets" / "logo" / "wildmatch-fullname.png"
BLUE = "#3a7eab"
RED = "#cf4832"
GREY = "#d1d3d4"
INK = "#58595b"
MUTED = "#6d6e71"
LINE = "#19c2d6"
DATASETS = ["CzechLynx", "Hyena", "Leopard", "Nyala", "Salamander", "Sea star", "Whale shark", "Turtle"]
FONT_KW = {"font": "DejaVu Sans"}

TIMINGS = json.loads((HERE / "audio" / "timings.json").read_text(encoding="utf-8"))
GAP = float(TIMINGS["gap_s"])
SHOTS = {s["shot"]: s for s in TIMINGS["shots"]}


def shot_end(shot: int) -> float:
    s = SHOTS[shot]
    return float(s["start"]) + float(s["duration"]) + GAP


def load_pair():
    data = json.loads((DEMO / "before_after" / "before_after.json").read_text(encoding="utf-8"))
    return next(p for p in data["pairs"] if p["dataset"] == "lynx_closed")


def load_anchor():
    data = json.loads((DEMO / "mined_pairs" / "mined_pairs.json").read_text(encoding="utf-8"))
    return data["anchors"][0]


def photo(path: Path, height: float) -> ImageMobject:
    img = ImageMobject(str(path))
    img.set(height=height)
    return img


def pixel_to_point(img: ImageMobject, xy, px_w: int, px_h: int):
    x, y = xy
    ul = img.get_corner(UL)
    return ul + RIGHT * (x / px_w) * img.width + DOWN * (y / px_h) * img.height


def label(text: str, size: float = 28, color: str = INK, **kw) -> Text:
    return Text(text, font_size=size, color=color, **FONT_KW, **kw)


class Explainer(Scene):
    def fill_to(self, t_end: float) -> None:
        remaining = t_end - self.renderer.time
        if remaining > 0.05:
            self.wait(remaining)

    def construct(self) -> None:
        self.camera.background_color = WHITE
        pair = load_pair()
        anchor = load_anchor()
        self.shot_1_2(pair)
        self.shot_3_4(anchor)
        self.shot_5()
        self.shot_6(pair)
        self.shot_7()

    # ------------------------------------------------------------------ shots 1 and 2
    def pair_images(self, pair, height=4.6):
        q = photo(DEMO / "before_after" / pair["query"]["image"]["file"], height)
        g = photo(DEMO / "before_after" / pair["gallery"]["image"]["file"], height)
        q.move_to(LEFT * 3.1 + UP * 0.3)
        g.move_to(RIGHT * 3.1 + UP * 0.3)
        return q, g

    def correspondences(self, pair, q, g, which: str, count: int, stroke: float):
        res = pair["results"][which]
        qp, gp = res["points"]["query"], res["points"]["gallery"]
        qw, qh = pair["query"]["image"]["width"], pair["query"]["image"]["height"]
        gw, gh = pair["gallery"]["image"]["width"], pair["gallery"]["image"]["height"]
        order = res["order_by_confidence"][:count]
        lines, dots = VGroup(), VGroup()
        for i in order:
            a = pixel_to_point(q, qp[i], qw, qh)
            b = pixel_to_point(g, gp[i], gw, gh)
            lines.add(Line(a, b, color=LINE, stroke_width=stroke, stroke_opacity=0.85))
            dots.add(Dot(a, radius=0.035, color=LINE), Dot(b, radius=0.035, color=LINE))
        return lines, dots

    def shot_1_2(self, pair) -> None:
        q, g = self.pair_images(pair)
        tag_q = label("query", 24, MUTED).next_to(q, DOWN, buff=0.15)
        tag_g = label("gallery photo", 24, MUTED).next_to(g, DOWN, buff=0.15)
        self.play(FadeIn(q), FadeIn(g), FadeIn(tag_q), FadeIn(tag_g), run_time=1.2)
        lines, dots = self.correspondences(pair, q, g, "default", 14, 2.5)
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in dots], lag_ratio=0.03), run_time=2.2)
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.08), run_time=2.6)
        title = label("A pretrained image matcher", 30, INK).to_edge(UP, buff=0.35)
        self.play(FadeIn(title), run_time=0.6)
        self.fill_to(shot_end(1))
        # shot 2: the matcher's training domain, and no labels to retrain it
        self.play(lines.animate.set_stroke(opacity=0.18), dots.animate.set_opacity(0.3), run_time=0.8)
        box = RoundedRectangle(corner_radius=0.15, width=9.6, height=0.9, color=GREY, fill_color="#f3f4f4", fill_opacity=1)
        box.to_edge(DOWN, buff=0.35)
        t1 = label("trained on buildings, streets and landmarks", 26, INK).move_to(box)
        self.play(FadeIn(box), Write(t1), run_time=1.4)
        t2 = label("wildlife photos: identity labels, no keypoint labels", 26, RED).move_to(box)
        self.wait(2.2)
        self.play(Transform(t1, t2), run_time=1.0)
        self.fill_to(shot_end(2) - 0.6)
        self.play(FadeOut(Group(q, g, tag_q, tag_g, lines, dots, title, box, t1)), run_time=0.6)

    # ------------------------------------------------------------------ shots 3 and 4
    def shot_3_4(self, anchor) -> None:
        folder = DEMO / "mined_pairs"
        records = [anchor["anchor"]] + anchor["positives"][:5] + anchor["negatives"][:4]
        identities = [anchor["identity"]] + [p["identity"] for p in anchor["positives"][:5]] + [n["identity"] for n in anchor["negatives"][:4]]
        imgs = Group(*[photo(folder / r["image"]["file"], 1.75) for r in records])
        imgs.arrange_in_grid(rows=2, cols=5, buff=0.35).move_to(UP * 0.1)
        tags = VGroup()
        for img, ident in zip(imgs, identities):
            name = ident.replace("lynx_", "lynx ")
            pill = RoundedRectangle(corner_radius=0.1, width=1.3, height=0.34, color=BLUE, fill_color=BLUE, fill_opacity=0.12, stroke_width=1.5)
            txt = label(name, 17, BLUE)
            pill.next_to(img, DOWN, buff=0.08)
            txt.move_to(pill)
            tags.add(VGroup(pill, txt))
        title = label("The reference database already names each animal", 30, INK).to_edge(UP, buff=0.35)
        self.play(FadeIn(title), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(i, shift=UP * 0.2) for i in imgs], lag_ratio=0.08), run_time=2.4)
        self.play(LaggedStart(*[FadeIn(t, scale=0.6) for t in tags], lag_ratio=0.06), run_time=1.6)
        self.fill_to(shot_end(3))
        # shot 4: anchor to the centre, positives left, negatives right
        title2 = label("The pretrained matcher scores the anchor against the rest", 30, INK).to_edge(UP, buff=0.35)
        self.play(Transform(title, title2), run_time=0.8)
        anchor_img, anchor_tag = imgs[0], tags[0]
        pos = list(zip(imgs[1:6], tags[1:6], anchor["positives"][:5]))
        neg = list(zip(imgs[6:10], tags[6:10], anchor["negatives"][:4]))
        anchor_target = anchor_img.copy().set(height=2.6).move_to(ORIGIN + DOWN * 0.1)
        moves = [anchor_img.animate.set(height=2.6).move_to(ORIGIN + DOWN * 0.1),
                 anchor_tag.animate.next_to(anchor_target, DOWN, buff=0.1)]
        def column(items, x, color):
            anims, frames, scores = [], VGroup(), VGroup()
            ys = np.linspace(2.0, -2.4, len(items))
            for (img, tag, rec), y in zip(items, ys):
                target = img.copy().set(height=1.0).move_to(RIGHT * x + UP * y)
                anims.append(img.animate.set(height=1.0).move_to(RIGHT * x + UP * y))
                anims.append(tag.animate.scale(0.75).next_to(target, DOWN if False else RIGHT * np.sign(x), buff=0.12))
                frames.add(SurroundingRectangle(target, color=color, buff=0.03, stroke_width=3))
                scores.add(label(f"{rec['mined_score']:.2f}", 18, color).next_to(target, LEFT * np.sign(x), buff=0.12))
            return anims, frames, scores
        pos_anims, pos_frames, pos_scores = column(pos, -4.6, BLUE)
        neg_anims, neg_frames, neg_scores = column(neg, 4.6, RED)
        self.play(*moves, *pos_anims, *neg_anims, run_time=2.2)
        pos_head = label("positives: same animal", 22, BLUE).move_to(LEFT * 4.6 + UP * 2.95)
        neg_head = label("hard negatives: other animals", 22, RED).move_to(RIGHT * 4.6 + UP * 2.95)
        self.play(FadeIn(pos_frames), FadeIn(pos_head), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(s) for s in pos_scores], lag_ratio=0.1), run_time=1.0)
        self.wait(1.2)
        self.play(FadeIn(neg_frames), FadeIn(neg_head), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(s) for s in neg_scores], lag_ratio=0.1), run_time=1.0)
        self.fill_to(shot_end(4) - 0.6)
        self.play(FadeOut(Group(imgs, tags, title, pos_frames, neg_frames, pos_scores, neg_scores, pos_head, neg_head)), run_time=0.6)

    # ------------------------------------------------------------------ shot 5
    def shot_5(self) -> None:
        title = label("Training asks for one thing", 30, INK).to_edge(UP, buff=0.35)
        axis = NumberLine(x_range=[0, 1, 0.25], length=9, color=GREY, include_numbers=False)
        axis.move_to(UP * 0.9)
        ticks = VGroup(*[label(f"{v:.2f}", 18, MUTED).next_to(axis.n2p(v), DOWN, buff=0.15) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        axis_label = VGroup(ticks, label("matcher score", 22, MUTED).next_to(ticks, DOWN, buff=0.15))
        p = Dot(axis.n2p(0.24), radius=0.13, color=BLUE)
        n = Dot(axis.n2p(0.16), radius=0.13, color=RED)
        p_tag = label("positive", 20, BLUE).next_to(p, UP, buff=0.15)
        n_tag = label("hard negative", 20, RED).next_to(n, UP, buff=0.15)
        self.play(FadeIn(title), Create(axis), FadeIn(axis_label), run_time=1.2)
        self.play(FadeIn(p, scale=0.5), FadeIn(n, scale=0.5), FadeIn(p_tag), FadeIn(n_tag), run_time=0.8)
        self.wait(0.8)
        p2, n2 = axis.n2p(0.72), axis.n2p(0.12)
        brace = Brace(Line(n2, p2), DOWN, buff=0.12, color=INK)
        brace_txt = label("margin", 22, INK).next_to(brace, DOWN, buff=0.1)
        self.play(p.animate.move_to(p2), n.animate.move_to(n2),
                  p_tag.animate.next_to(p2, UP, buff=0.15), n_tag.animate.next_to(n2, UP, buff=0.15),
                  run_time=1.6, rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeIn(brace), FadeIn(brace_txt), FadeOut(axis_label), run_time=0.8)
        # the three blocks: only the matching module changes
        names = ["keypoint detector", "descriptor", "matching module"]
        blocks = VGroup()
        for name in names:
            box = RoundedRectangle(corner_radius=0.12, width=3.6, height=0.9, color=GREY, fill_color="#f3f4f4", fill_opacity=1, stroke_width=2)
            blocks.add(VGroup(box, label(name, 22, MUTED).move_to(box)))
        blocks.arrange(RIGHT, buff=0.4).move_to(DOWN * 2.2)
        frozen = VGroup(*[label("frozen", 18, MUTED).next_to(b, DOWN, buff=0.1) for b in blocks[:2]])
        self.wait(1.0)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in blocks], lag_ratio=0.15), run_time=1.2)
        trained = label("trained", 18, BLUE).next_to(blocks[2], DOWN, buff=0.1)
        self.play(blocks[2][0].animate.set_stroke(color=BLUE, width=4).set_fill("#cedfea"),
                  blocks[2][1].animate.set_color(BLUE), FadeIn(trained), FadeIn(frozen), run_time=1.0)
        self.fill_to(shot_end(5) - 0.6)
        self.play(FadeOut(VGroup(title, axis, p, n, p_tag, n_tag, brace, brace_txt, blocks, trained, frozen)), run_time=0.6)

    # ------------------------------------------------------------------ shot 6
    def shot_6(self, pair) -> None:
        q, g = self.pair_images(pair, height=4.2)
        q.move_to(LEFT * 4.4 + UP * 0.2)
        g.move_to(LEFT * 0.0 + UP * 0.2)
        title = label("After adaptation", 30, INK).to_edge(UP, buff=0.35)
        self.play(FadeIn(q), FadeIn(g), FadeIn(title), run_time=0.8)
        lines, dots = self.correspondences(pair, q, g, "finetuned", 60, 1.8)
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in dots], lag_ratio=0.01), run_time=1.2)
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.02), run_time=2.0)
        count = label(f"{pair['results']['finetuned']['match_count']} matches, score {pair['results']['finetuned']['score']:.2f}", 22, BLUE)
        count.next_to(g, DOWN, buff=0.15)
        before = label(f"before: {pair['results']['default']['match_count']} matches, score {pair['results']['default']['score']:.2f}", 20, MUTED)
        before.next_to(count, DOWN, buff=0.08)
        self.play(FadeIn(count), FadeIn(before), run_time=0.6)
        # candidate list: the real top-5 of this query under both matchers (candidates.json)
        cand = json.loads((HERE / "candidates.json").read_text(encoding="utf-8"))["methods"]
        def rows_for(method):
            rows = VGroup()
            for entry in cand[method]["top5"]:
                color = BLUE if entry["correct"] else MUTED
                box = RoundedRectangle(corner_radius=0.08, width=3.2, height=0.55, color=BLUE if entry["correct"] else GREY,
                                       fill_color="#cedfea" if entry["correct"] else "#f7f7f7", fill_opacity=1, stroke_width=2.5 if entry["correct"] else 1.5)
                name = label(entry["identity"].replace("lynx_", "lynx "), 20, color).move_to(box).shift(LEFT * 0.55)
                score = label(f"{entry['score']:.2f}", 18, color).move_to(box).shift(RIGHT * 1.05)
                rows.add(VGroup(box, name, score))
            rows.arrange(DOWN, buff=0.12).move_to(RIGHT * 4.7 + UP * 0.1)
            return rows
        rows = rows_for("default")
        head = label("candidates: default matcher", 20, MUTED).next_to(rows, UP, buff=0.2)
        self.play(FadeIn(head), LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows], lag_ratio=0.1), run_time=1.0)
        self.wait(0.6)
        rows2 = rows_for("finetuned")
        head2 = label("candidates: after adaptation", 20, BLUE).next_to(rows2, UP, buff=0.2)
        self.play(Transform(rows, rows2), Transform(head, head2), run_time=1.2, rate_func=rate_functions.ease_in_out_sine)
        self.fill_to(shot_end(6) - 0.6)
        self.play(FadeOut(Group(q, g, title, lines, dots, count, before, rows, head)), run_time=0.6)

    # ------------------------------------------------------------------ shot 7
    def shot_7(self) -> None:
        cells = VGroup()
        for name in DATASETS:
            arrow = Arrow(DOWN * 0.35, UP * 0.35, color=BLUE, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.35)
            txt = label(name, 20, INK)
            cells.add(VGroup(arrow, txt).arrange(DOWN, buff=0.12))
        cells.arrange_in_grid(rows=2, cols=4, buff=(0.9, 0.5)).move_to(UP * 1.1)
        head = label("Eight wildlife datasets", 26, MUTED).next_to(cells, UP, buff=0.4)
        self.play(FadeIn(head), LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cells], lag_ratio=0.08), run_time=2.0)
        cost = label("about 5 GPU-hours of training", 24, INK).move_to(DOWN * 1.5)
        self.play(FadeIn(cost), run_time=0.8)
        self.wait(1.4)
        logo = ImageMobject(str(LOGO)).set(height=3.6)
        self.play(FadeOut(cost), FadeOut(cells), FadeOut(head), run_time=0.6)
        logo.move_to(ORIGIN)
        self.play(FadeIn(logo, scale=0.95), run_time=1.0)
        self.fill_to(shot_end(7) + 0.4)
