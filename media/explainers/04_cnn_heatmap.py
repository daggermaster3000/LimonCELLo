"""04 — CNN ROI classifier, then the fully-convolutional heatmap detector."""
from manim import *
from shared import *


def conv_stack(color=ACCENT):
    """Three offset squares standing in for stacked convolution layers."""
    g = VGroup()
    for i in range(3):
        s = Square(side_length=0.9, stroke_width=2, stroke_color=color,
                   fill_color=BG, fill_opacity=0.85)
        s.shift(RIGHT * 0.18 * i + UP * 0.18 * i)
        g.add(s)
    return g


def heat_grid(centers, rows=9, cols=14, cell=0.42, sigma=1.1):
    g = VGroup()
    vals = {}
    for r in range(rows):
        for c in range(cols):
            v = 0.0
            for (cx, cy) in centers:
                v += np.exp(-(((c - cx) ** 2 + (r - cy) ** 2) / (2 * sigma ** 2)))
            v = min(v, 1.0)
            vals[(r, c)] = v
            sq = Square(side_length=cell, stroke_width=0.4, stroke_color="#1c1030",
                        fill_color=interpolate_color(ManimColor("#13071f"), ManimColor(ACCENT), v),
                        fill_opacity=0.25 + 0.7 * v)
            sq.move_to([(c - (cols - 1) / 2) * cell, ((rows - 1) / 2 - r) * cell, 0])
            g.add(sq)
    g.vals = vals
    g.rows, g.cols, g.cell = rows, cols, cell
    return g


class CnnAndHeatmap(Scene):
    def construct(self):
        setup_scene(self)
        cap = Caption(self)

        title_card(self, "Step 3 · Classify",
                   "a learned filter:  cilium or junk?")

        # ── PART A: one ROI through the CNN ──────────────────────────
        roi = VGroup(
            Square(side_length=2.0, stroke_width=2.5, stroke_color=CILIA,
                   fill_color=BG, fill_opacity=0.9),
            cilium_glyph(scale=1.1),
        ).move_to(LEFT * 4.2)
        roi_lbl = Text("one ROI crop", font_size=22, color=GREY_B).next_to(roi, DOWN, buff=0.3)
        self.play(FadeIn(roi, scale=0.8), FadeIn(roi_lbl), run_time=0.7)
        cap.show("For each candidate we cut out a small crop around it.", 2.4)

        cnn = conv_stack().move_to(LEFT * 0.6)
        cnn_lbl = Text("conv-net", font_size=22, color=ACCENT).next_to(cnn, DOWN, buff=0.5)
        a1 = Arrow(roi.get_right(), cnn.get_left(), buff=0.25, stroke_width=4, color=GREY_B)
        self.play(GrowArrow(a1), Create(cnn), FadeIn(cnn_lbl), run_time=0.9)
        cap.show("A small neural net has learned what a real cilium looks like.", 2.6)

        # score bar junk↔cilium
        bar = NumberLine(x_range=[0, 1, 0.5], length=3.2, include_numbers=False,
                         color=GREY_B).move_to(RIGHT * 3.6 + UP * 0.2)
        junk = Text("junk", font_size=20, color=JUNK).next_to(bar, LEFT, buff=0.2)
        good = Text("cilium", font_size=20, color=CILIA).next_to(bar, RIGHT, buff=0.2)
        ptr = Triangle(color=ACCENT, fill_opacity=1).scale(0.14).rotate(PI)
        ptr.next_to(bar.n2p(0.5), UP, buff=0.05)
        a2 = Arrow(cnn.get_right(), bar.get_left() + LEFT * 0.6, buff=0.2,
                   stroke_width=4, color=GREY_B)
        self.play(GrowArrow(a2), Create(bar), FadeIn(junk), FadeIn(good), FadeIn(ptr), run_time=0.9)
        self.play(ptr.animate.next_to(bar.n2p(0.9), UP, buff=0.05), run_time=1.0)
        score = Text("score 0.9 → keep", font_size=22, color=CILIA).next_to(bar, DOWN, buff=0.5)
        self.play(FadeIn(score, shift=UP * 0.2), run_time=0.4)
        cap.show("It scores the crop from junk to cilium. High score → keep.", 2.6)

        partA = VGroup(roi, roi_lbl, cnn, cnn_lbl, a1, a2, bar, junk, good, ptr, score)
        self.play(FadeOut(partA), run_time=0.6)
        cap.clear()

        # ── PART B: fully-convolutional sweep ────────────────────────
        sec = Text("but the whole image has thousands of spots…",
                   font_size=28, color=ACCENT).to_edge(UP, buff=0.7)
        fit_width(sec, 12)
        self.play(FadeIn(sec, shift=DOWN * 0.2), run_time=0.5)

        base = grid_image(rows=9, cols=14, cell=0.42).move_to(DOWN * 0.2)
        baseframe = SurroundingRectangle(base, color=GREY_B, buff=0.04, stroke_width=2)
        self.play(Create(baseframe), FadeIn(base), run_time=0.7)

        # naive: a window slides crop by crop
        win = Square(side_length=0.42 * 3, stroke_width=3, stroke_color=JUNK)
        win.move_to(base[0].get_center() + RIGHT * 0.42 + DOWN * 0.42)
        self.play(FadeIn(win), run_time=0.3)
        path = [win.get_center() + RIGHT * 0.42 * k for k in range(0, 9, 2)]
        self.play(Succession(*[win.animate(run_time=0.18).move_to(p) for p in path]))
        cap.show("Scoring one crop at a time would be far too slow.", 2.4)

        # the trick: run the net once, everywhere
        self.play(FadeOut(win), run_time=0.3)
        sec2 = Text("run the same net once, everywhere → a heatmap",
                    font_size=26, color=ACCENT)
        fit_width(sec2, 12)
        sec2.move_to(sec)
        self.play(Transform(sec, sec2), run_time=0.6)

        centers = [(3, 2), (5, 6), (8, 3), (10, 6), (12, 2)]  # (col, row)
        heat = heat_grid([(cx, cy) for (cx, cy) in centers]).move_to(base.get_center())
        self.play(FadeOut(base), FadeIn(heat), run_time=1.0)
        cap.show("The net lights up wherever it sees a real cilium.", 2.6)

        # peak-find → crosses at the centres
        crosses = VGroup()
        for (cx, cy) in centers:
            pos = [(cx - (heat.cols - 1) / 2) * heat.cell,
                   ((heat.rows - 1) / 2 - cy) * heat.cell, 0]
            cr = VGroup(
                Line(LEFT * 0.16, RIGHT * 0.16, stroke_width=4, color=INK),
                Line(DOWN * 0.16, UP * 0.16, stroke_width=4, color=INK),
            ).move_to(pos)
            crosses.add(cr)
        self.play(LaggedStart(*[GrowFromCenter(c) for c in crosses], lag_ratio=0.2), run_time=1.2)
        cap.show("We pick the peaks — one mark per cilium.", 2.4)

        # hand back ROIs
        rings = VGroup(*[Circle(radius=0.55, stroke_width=3, stroke_color=CILIA).move_to(c)
                         for c in crosses])
        self.play(LaggedStartMap(Create, rings, lag_ratio=0.2), run_time=1.0)
        cap.show("Each peak becomes an ROI — back to the measuring step.", 2.6)
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
