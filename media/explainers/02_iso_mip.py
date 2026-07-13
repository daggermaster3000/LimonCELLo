"""02 — Flatten: isotropic resampling + maximum-intensity projection."""
from manim import *
from shared import *


class IsoAndMIP(Scene):
    def construct(self):
        setup_scene(self)
        cap = Caption(self)

        title_card(self, "Step 1 · Flatten the stack",
                   "isotropic resampling  +  max projection")

        # ────────────────────────────────────────────────────────────
        # PART A — anisotropic voxels are not cubes
        # ────────────────────────────────────────────────────────────
        secA = Text("a voxel is not a cube", font_size=30, color=ACCENT).to_edge(UP, buff=0.6)
        self.play(FadeIn(secA, shift=DOWN * 0.2), run_time=0.5)

        # z-x cross-section: tall (anisotropic) cells
        ncol, nrow = 5, 3
        cw, ch = 0.62, 1.18
        aniso = VGroup()
        for r in range(nrow):
            for c in range(ncol):
                cell = Rectangle(width=cw, height=ch, stroke_width=1.5,
                                 stroke_color=NUCLEI, fill_color=NUCLEI, fill_opacity=0.12)
                cell.move_to([(c - (ncol - 1) / 2) * cw,
                              ((nrow - 1) / 2 - r) * ch, 0])
                aniso.add(cell)
        aniso.move_to(ORIGIN + DOWN * 0.3)
        xax = Arrow(aniso.get_corner(DL) + LEFT * 0.1, aniso.get_corner(DR) + RIGHT * 0.4,
                    buff=0, stroke_width=3, color=GREY_B)
        zax = Arrow(aniso.get_corner(DL) + DOWN * 0.1, aniso.get_corner(UL) + UP * 0.4,
                    buff=0, stroke_width=3, color=GREY_B)
        xlbl = Text("x, y", font_size=22, color=GREY_B).next_to(xax, RIGHT, buff=0.1)
        zlbl = Text("z", font_size=22, color=GREY_B).next_to(zax, UP, buff=0.1)

        self.play(LaggedStartMap(FadeIn, aniso, lag_ratio=0.03), run_time=1.2)
        self.play(GrowArrow(xax), GrowArrow(zax), FadeIn(xlbl), FadeIn(zlbl), run_time=0.6)
        cap.show("Slices are spaced wider in z than pixels are in x and y.", 2.6)

        # resample → cubes
        iso = VGroup()
        s = 0.78
        for r in range(nrow):
            for c in range(ncol):
                cell = Square(side_length=s, stroke_width=1.5,
                              stroke_color=CILIA, fill_color=CILIA, fill_opacity=0.12)
                cell.move_to([(c - (ncol - 1) / 2) * s,
                              ((nrow - 1) / 2 - r) * s, 0])
                iso.add(cell)
        iso.move_to(aniso.get_center())
        cap.show("So we resample the stack until every voxel is a cube.", 2.4)
        self.play(Transform(aniso, iso), run_time=1.4)
        eq = Text("now 1 voxel = real micrometres in every direction",
                  font_size=24, color=INK).next_to(aniso, DOWN, buff=0.5)
        fit_width(eq, 12)
        self.play(FadeIn(eq, shift=UP * 0.2), run_time=0.5)
        self.wait(0.6)

        self.play(FadeOut(VGroup(secA, aniso, xax, zax, xlbl, zlbl, eq)), run_time=0.6)
        cap.clear()

        # ────────────────────────────────────────────────────────────
        # PART B — maximum intensity projection
        # ────────────────────────────────────────────────────────────
        secB = Text("max-intensity projection", font_size=30, color=ACCENT).to_edge(UP, buff=0.6)
        self.play(FadeIn(secB, shift=DOWN * 0.2), run_time=0.5)

        # a deck of slices, each with a few bright signal dots
        rng = np.random.default_rng(3)
        n_sl = 4
        slices = VGroup()
        skew = RIGHT * 1.05 + UP * 0.62
        dot_sets = []
        palette = [CILIA, BB, CILIA, NUCLEI]
        for i in range(n_sl):
            plane = Rectangle(width=3.0, height=2.2, stroke_width=2,
                              stroke_color=GREY_B, fill_color=BG, fill_opacity=0.65)
            plane.shift(skew * i * 0.5 + LEFT * 2.6 + DOWN * 0.2)
            dots = VGroup()
            for _ in range(2):
                px = rng.uniform(-1.2, 1.2)
                py = rng.uniform(-0.8, 0.8)
                d = Dot(radius=0.12, color=palette[i]).move_to(plane.get_center() + [px, py, 0])
                d.base_xy = (px, py)
                dots.add(d)
            dot_sets.append(dots)
            slices.add(VGroup(plane, dots))

        self.play(LaggedStartMap(FadeIn, slices, lag_ratio=0.25), run_time=1.4)
        cap.show("After flattening we have many 2-D slices, one behind another.", 2.6)

        # the projection target plane (front)
        target = Rectangle(width=3.0, height=2.2, stroke_width=2.5,
                           stroke_color=INK, fill_color=BG, fill_opacity=0.9)
        target.move_to(RIGHT * 3.4 + DOWN * 0.2)
        tlbl = Text("flat image (MIP)", font_size=22, color=INK).next_to(target, DOWN, buff=0.3)
        arrow = Arrow(slices.get_right(), target.get_left(), buff=0.3,
                      stroke_width=4, color=ACCENT)
        max_lbl = Text("keep the brightest", font_size=20, color=ACCENT).next_to(arrow, UP, buff=0.12)
        self.play(Create(target), FadeIn(tlbl), run_time=0.6)
        self.play(GrowArrow(arrow), FadeIn(max_lbl), run_time=0.5)
        cap.show("Down each column we keep only the brightest pixel.", 2.6)

        # collapse: every dot flies onto the target plane at its x,y
        flights = []
        merged = VGroup()
        for dots in dot_sets:
            for d in dots:
                px, py = d.base_xy
                dest = target.get_center() + np.array([px, py, 0])
                dc = d.copy()
                flights.append(dc.animate.move_to(dest))
                merged.add(dc)
        self.add(merged)
        self.play(*flights, run_time=1.6)
        self.play(Flash(target, color=ACCENT, line_length=0.25, num_lines=16), run_time=0.6)
        cap.show("One sharp 2-D image — every cilium visible at once.", 2.6)
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
