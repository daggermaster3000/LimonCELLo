"""03 — APOC pixel classifier: per-pixel features + a forest of tiny trees."""
from manim import *
from shared import *


class ApocPixelClassifier(Scene):
    def construct(self):
        setup_scene(self)
        cap = Caption(self)

        title_card(self, "Step 2 · Segment",
                   "ask every pixel a few questions")

        # ── the image + one chosen pixel ─────────────────────────────
        img = grid_image(rows=8, cols=8, cell=0.5).move_to(LEFT * 3.6)
        frame = SurroundingRectangle(img, color=GREY_B, buff=0.05, stroke_width=2)
        # sprinkle a faint cilium so the image means something
        cil = cilium_glyph(scale=1.0).move_to(img.get_center() + UR * 0.6).set_opacity(0.5)
        self.play(Create(frame), LaggedStartMap(Create, img, lag_ratio=0.01), run_time=1.4)
        self.play(FadeIn(cil), run_time=0.4)
        cap.show("Segmentation = decide, for every pixel: object or background?", 2.6)

        # highlight one pixel
        target_cell = img[3 * 8 + 5]
        hl = target_cell.copy().set_stroke(ACCENT, width=4).set_fill(ACCENT, 0.25)
        self.play(FadeIn(hl, scale=1.4), run_time=0.5)
        cap.show("Take one pixel. By itself, its brightness tells us little.", 2.4)

        # ── feature fan-out ──────────────────────────────────────────
        feats = ["brightness", "edges", "blur · small", "blur · big", "texture"]
        fcolors = [INK, ACCENT, NUCLEI, NUCLEI, BB]
        rows = VGroup()
        for name, col in zip(feats, fcolors):
            chip = RoundedRectangle(corner_radius=0.1, width=3.0, height=0.62,
                                    stroke_width=2, stroke_color=col, fill_opacity=0.0)
            t = Text(name, font_size=22, color=col).move_to(chip)
            rows.add(VGroup(chip, t))
        rows.arrange(DOWN, buff=0.22).move_to(RIGHT * 0.2 + UP * 0.2)
        flbl = Text("describe its neighbourhood", font_size=22, color=GREY_B)
        flbl.next_to(rows, UP, buff=0.3)

        fan = VGroup(*[Arrow(hl.get_right(), r.get_left(), buff=0.15,
                             stroke_width=2.5, color=GREY_B) for r in rows])
        self.play(FadeIn(flbl, shift=UP * 0.2), run_time=0.4)
        self.play(LaggedStart(*[GrowArrow(a) for a in fan], lag_ratio=0.15),
                  LaggedStartMap(FadeIn, rows, lag_ratio=0.15), run_time=1.6)
        cap.show("Instead we describe its neighbourhood: edges, blur, texture…", 2.8)

        # ── forest of tiny trees votes ───────────────────────────────
        self.play(VGroup(flbl, fan).animate.set_opacity(0.0),
                  rows.animate.scale(0.7).to_edge(UP, buff=0.7).set_x(0.2), run_time=0.8)
        forest = VGroup(*[tiny_tree(scale=0.55) for _ in range(5)])
        forest.arrange(RIGHT, buff=0.55).move_to(DOWN * 0.4 + RIGHT * 0.2)
        flabel = Text("a forest of tiny decision trees", font_size=22, color=GREY_B)
        flabel.next_to(forest, UP, buff=0.3)
        self.play(LaggedStart(*[GrowFromCenter(t) for t in forest], lag_ratio=0.15),
                  FadeIn(flabel), run_time=1.3)
        cap.show("A forest of tiny trees each votes: object, or background?", 2.6)

        votes = VGroup()
        for tr in forest:
            v = Text("object", font_size=18, color=CILIA).next_to(tr, DOWN, buff=0.2)
            votes.add(v)
        votes[2].become(Text("backgr.", font_size=18, color=GREY_B).move_to(votes[2]))
        self.play(LaggedStartMap(FadeIn, votes, lag_ratio=0.12), run_time=1.0)
        tally = Text("majority → OBJECT", font_size=26, color=ACCENT, weight=BOLD)
        tally.next_to(forest, DOWN, buff=1.0)
        self.play(FadeIn(tally, shift=UP * 0.2), run_time=0.5)
        cap.show("The majority wins. This pixel is part of a cilium.", 2.4)

        # ── paint the masks ──────────────────────────────────────────
        self.play(FadeOut(VGroup(rows, forest, flabel, votes, tally, hl)), run_time=0.6)
        self.play(img.animate.move_to(ORIGIN).scale(1.05),
                  frame.animate.move_to(ORIGIN).scale(1.05),
                  cil.animate.move_to(UR * 0.65).set_opacity(0.7), run_time=0.8)
        cap.show("Do this for every pixel and the masks paint themselves.", 2.6)

        rng = np.random.default_rng(7)
        cilia_px = VGroup()
        bb_px = VGroup()
        center = img.get_center()
        for cell in img:
            p = cell.get_center()
            d = np.linalg.norm(p - (center + UR * 0.65))
            if d < 0.9:
                cilia_px.add(cell.copy().set_fill(CILIA, 0.7).set_stroke(width=0))
            elif d < 1.15 and rng.random() < 0.5:
                bb_px.add(cell.copy().set_fill(BB, 0.85).set_stroke(width=0))
        self.play(LaggedStartMap(FadeIn, cilia_px, lag_ratio=0.04), run_time=1.2)
        self.play(LaggedStartMap(FadeIn, bb_px, lag_ratio=0.06), run_time=0.8)

        legend = VGroup(
            VGroup(Dot(color=CILIA), Text("cilia", font_size=22, color=INK)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(color=BB), Text("basal bodies", font_size=22, color=INK)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_edge(RIGHT, buff=0.8)
        self.play(FadeIn(legend, shift=LEFT * 0.2), run_time=0.6)
        cap.show("Two masks: green cilia and their magenta basal bodies.", 2.6)
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
