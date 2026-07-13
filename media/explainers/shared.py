"""Shared palette, glyphs and caption helper for the LimonCELLo explainer videos.

Keeps the 4 clips looking like one set. No LaTeX is used anywhere (Text only) so the
toolchain needs ffmpeg but NOT a TeX install.
"""
from manim import *

# ── Palette (matches the napari/data_app channel colours) ─────────────────────
BG      = "#0b0221"   # deep indigo background
INK     = "#eaeaf0"   # near-white text
CILIA   = "#39ff14"   # neon green  — cilia channel
BB      = "#ff5bd1"   # magenta     — basal bodies
NUCLEI  = "#4ea3ff"   # blue        — nuclei
NEURITE = "#9aa0a6"   # grey        — neurites / soma
ACCENT  = "#ffd23f"   # amber       — highlights / scores
JUNK    = "#ff4d4d"   # red         — rejected / junk


def setup_scene(scene):
    """Apply the house background colour."""
    scene.camera.background_color = BG


def fit_width(mobj, w):
    """Scale a mobject down (never up) so it is at most ``w`` units wide."""
    if mobj.width > w:
        mobj.scale(w / mobj.width)
    return mobj


class Caption:
    """A persistent bottom caption that also emits .srt subcaptions.

    Usage:
        cap = Caption(self)
        cap.show("first beat", t=2)
        cap.show("second beat", t=2.5)
        cap.clear()
    """

    def __init__(self, scene, font_size=30):
        self.scene = scene
        self.font_size = font_size
        self.txt = None

    def show(self, s, t=2.0):
        new = Text(s, font_size=self.font_size, color=INK)
        fit_width(new, 12.5)
        new.to_edge(DOWN, buff=0.35)
        self.scene.add_subcaption(s, duration=t)
        if self.txt is None:
            self.scene.play(FadeIn(new, shift=UP * 0.2), run_time=0.4)
        else:
            self.scene.play(
                FadeOut(self.txt, shift=UP * 0.2),
                FadeIn(new, shift=UP * 0.2),
                run_time=0.4,
            )
        self.txt = new
        self.scene.wait(t)
        return new

    def clear(self):
        if self.txt is not None:
            self.scene.play(FadeOut(self.txt), run_time=0.3)
            self.txt = None


def title_card(scene, line1, line2, accent=ACCENT):
    """Standard opening card: big title + subtitle, fade in then out."""
    t1 = Text(line1, font_size=58, color=INK, weight=BOLD)
    t2 = Text(line2, font_size=30, color=accent)
    fit_width(t1, 12)
    fit_width(t2, 11)
    g = VGroup(t1, t2).arrange(DOWN, buff=0.35).move_to(ORIGIN)
    scene.play(Write(t1), run_time=0.9)
    scene.play(FadeIn(t2, shift=UP * 0.2), run_time=0.6)
    scene.wait(0.8)
    scene.play(FadeOut(g), run_time=0.5)


# ── Reusable glyphs ───────────────────────────────────────────────────────────

def microscope_stack(n=5, w=2.6, h=1.7, gap=0.16, color=NUCLEI):
    """A small isometric z-stack of slices (suggests a 3-D confocal volume)."""
    slices = VGroup()
    skew = RIGHT * 0.5 + UP * 0.34
    for i in range(n):
        r = Rectangle(width=w, height=h, stroke_width=2,
                      stroke_color=color, fill_color=BG, fill_opacity=0.85)
        r.shift(skew * i * gap * 6)
        slices.add(r)
    slices.set_z_index(0)
    # brighten the top slice
    slices[-1].set_stroke(INK, width=2.5)
    return slices


def cilium_glyph(color=CILIA, bb_color=BB, scale=1.0, wobble=0.0):
    """A cilium: a curved tail growing out of a basal-body dot."""
    base = Dot(radius=0.10 * scale, color=bb_color)
    tail = ParametricFunction(
        lambda t: np.array([
            0.10 * scale + 0.9 * scale * t,
            0.55 * scale * t * t + wobble * np.sin(6 * t) * 0.15,
            0,
        ]),
        t_range=[0, 1],
        stroke_width=5,
        color=color,
    )
    g = VGroup(tail, base)
    return g


def grid_image(rows=6, cols=6, cell=0.42, stroke=GREY_D):
    """An empty pixel grid used to stand in for an image."""
    sq = VGroup()
    for r in range(rows):
        for c in range(cols):
            s = Square(side_length=cell, stroke_width=1.0, stroke_color=stroke,
                       fill_opacity=0.0)
            s.move_to(np.array([(c - (cols - 1) / 2) * cell,
                                ((rows - 1) / 2 - r) * cell, 0]))
            sq.add(s)
    return sq


def tiny_tree(color=INK, scale=0.5):
    """A minimal 3-node decision tree glyph for the random-forest panel."""
    root = Dot(radius=0.07, color=color)
    l = Dot(radius=0.07, color=color).shift(DL * 0.5 * scale + DOWN * 0.2)
    r = Dot(radius=0.07, color=color).shift(DR * 0.5 * scale + DOWN * 0.2)
    e1 = Line(root.get_center(), l.get_center(), stroke_width=2, color=color)
    e2 = Line(root.get_center(), r.get_center(), stroke_width=2, color=color)
    return VGroup(e1, e2, root, l, r).scale(scale)
