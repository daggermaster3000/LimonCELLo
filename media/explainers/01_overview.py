"""01 — Pipeline overview: the real LimonCELLo DAG (two parallel branches)."""
from manim import *
from shared import *

BLURPLE = "#5b6ef5"


def node(label, color=BLURPLE, w=1.62, h=1.18, fs=15, fill=0.18, text_color=INK):
    box = RoundedRectangle(corner_radius=0.1, width=w, height=h, stroke_width=2,
                           stroke_color=color, fill_color=color, fill_opacity=fill)
    t = Text(label, font_size=fs, color=text_color, weight=BOLD,
             line_spacing=0.7).move_to(box)
    fit_width(t, w - 0.2)
    g = VGroup(box, t)
    g.box = box
    return g


def split_node(label, c1, c2, w=1.62, h=1.18, fs=15):
    left = Rectangle(width=w / 2, height=h, stroke_width=0, fill_color=c1, fill_opacity=1)
    right = Rectangle(width=w / 2, height=h, stroke_width=0, fill_color=c2, fill_opacity=1)
    right.next_to(left, RIGHT, buff=0)
    body = VGroup(left, right)
    outline = RoundedRectangle(corner_radius=0.1, width=w, height=h,
                               stroke_width=2, stroke_color=INK).move_to(body)
    t = Text(label, font_size=fs, color=WHITE, weight=BOLD, line_spacing=0.7).move_to(body)
    fit_width(t, w - 0.2)
    g = VGroup(body, outline, t)
    g.box = outline
    return g


def arr(a, b, color=GREY_B):
    return Arrow(a, b, buff=0.12, stroke_width=4, color=color,
                 max_tip_length_to_length_ratio=0.22, max_stroke_width_to_length_ratio=8)


class PipelineOverview(Scene):
    def construct(self):
        setup_scene(self)
        cap = Caption(self)

        title_card(self, "LimonCELLo", "how the pipeline finds & measures cilia")

        yT, yB = 1.05, -1.75
        x = [-6.0, -4.05, -2.1, -0.15, 1.8, 3.75, 5.6]

        raw   = node("RAW\nimages", NUCLEI, fill=0.10).move_to([x[0], yT, 0])
        pre   = node("Rescale\nFilter\nBG removal").move_to([x[1], yT, 0])
        pix   = split_node("Pixel\nclassifier", CILIA, BB).move_to([x[2], yT, 0])
        bba   = node("BB assign\n+ size filter").move_to([x[3], yT, 0])
        mapbb = node("Map BB\n→ neurite").move_to([x[4], yT, 0])
        cnn   = node("CNN\nvalidation", ACCENT, fill=0.16).move_to([x[5], yT, 0])
        down  = node("Downstream\nanalysis", BB, fill=0.16).move_to([x[6], yT, 0])

        vor   = split_node("Voronoi-\nOtsu", "#19e0d0", BLURPLE).move_to([x[2], yB, 0])
        skel  = node("Skeleton\n+ distance", "#19e0d0").move_to([x[3], yB, 0])

        # ── build the top (cilia) lane ───────────────────────────────
        self.play(FadeIn(raw, scale=0.85), run_time=0.5)
        cap.show("We start with raw 3-D confocal images of neurons.", 2.2)
        self.play(GrowArrow(arr(raw.box.get_right(), pre.box.get_left())),
                  FadeIn(pre, shift=RIGHT * 0.2), run_time=0.6)
        cap.show("First, clean-up: rescale, filter, remove background.", 2.3)

        cap.show("From here, two jobs run in parallel.", 1.9)
        # cilia branch
        self.play(GrowArrow(arr(pre.box.get_right(), pix.box.get_left())),
                  FadeIn(pix, shift=RIGHT * 0.2), run_time=0.6)
        ut1 = Text("user-trained", font_size=18, color=ACCENT).next_to(pix, UP, buff=0.18)
        self.play(FadeIn(ut1, shift=DOWN * 0.1), run_time=0.4)
        cap.show("A user-trained pixel classifier paints cilia & basal bodies.", 2.6)
        self.play(GrowArrow(arr(pix.box.get_right(), bba.box.get_left())),
                  FadeIn(bba, shift=RIGHT * 0.2), run_time=0.6)
        cap.show("Basal bodies get assigned and filtered by size.", 2.2)

        # ── neurite branch ───────────────────────────────────────────
        self.play(GrowArrow(arr(pre.box.get_bottom(), vor.box.get_left())),
                  FadeIn(vor, shift=RIGHT * 0.2), run_time=0.6)
        cap.show("In parallel, Voronoi-Otsu segments the neurites.", 2.3)
        self.play(GrowArrow(arr(vor.box.get_right(), skel.box.get_left())),
                  FadeIn(skel, shift=RIGHT * 0.2), run_time=0.6)
        cap.show("Their skeleton and distance maps describe the network.", 2.4)

        # ── merge → map BB onto neurite ──────────────────────────────
        self.play(GrowArrow(arr(bba.box.get_right(), mapbb.box.get_left())),
                  GrowArrow(arr(skel.box.get_right(), mapbb.box.get_bottom())),
                  FadeIn(mapbb, shift=UP * 0.2), run_time=0.8)
        cap.show("Both branches meet: each basal body is placed on its neurite.", 2.7)

        # ── CNN validation + downstream ──────────────────────────────
        self.play(GrowArrow(arr(mapbb.box.get_right(), cnn.box.get_left())),
                  FadeIn(cnn, shift=RIGHT * 0.2), run_time=0.6)
        ut2 = Text("user-trained", font_size=18, color=ACCENT).next_to(cnn, UP, buff=0.18)
        self.play(FadeIn(ut2, shift=DOWN * 0.1), run_time=0.4)
        cap.show("A user-trained CNN keeps real cilia and drops the junk.", 2.5)
        self.play(GrowArrow(arr(cnn.box.get_right(), down.box.get_left())),
                  FadeIn(down, shift=RIGHT * 0.2), run_time=0.6)
        cap.show("What survives feeds the downstream data analysis.", 2.3)

        # ── scope brackets ───────────────────────────────────────────
        cap.clear()
        core = VGroup(raw, pre, pix, bba, vor, skel, mapbb)
        appg = VGroup(cnn, down)
        b1 = Brace(core, DOWN, color=GREY_B)
        b1l = Text("LimonCELLo pipeline", font_size=24, color=INK).next_to(b1, DOWN, buff=0.12)
        b2 = Brace(appg, DOWN, color=GREY_B)
        b2l = Text("Streamlit data app", font_size=24, color=INK).next_to(b2, DOWN, buff=0.12)
        self.play(GrowFromCenter(b1), GrowFromCenter(b2), run_time=0.7)
        self.play(FadeIn(b1l), FadeIn(b2l), run_time=0.5)
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
