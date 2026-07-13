# LimonCELLo explainer videos

Lay-audience animations of the analysis pipeline, built with
[manim](https://www.manim.community/) (Community Edition) + the
[manim-skill](https://github.com/Yusuke710/manim-skill) workflow. On-screen
captions per beat; a matching `.srt` is emitted next to every `.mp4`.

| # | File | Scene class | Covers |
|---|------|-------------|--------|
| 01 | `01_overview.py` | `PipelineOverview` | The whole journey: stack → flatten → segment → classify → measure |
| 02 | `02_iso_mip.py` | `IsoAndMIP` | Isotropic resampling (cubic voxels) + max-intensity projection |
| 03 | `03_apoc.py` | `ApocPixelClassifier` | Per-pixel features → random-forest vote → cilia / basal-body masks |
| 04 | `04_cnn_heatmap.py` | `CnnAndHeatmap` | CNN cilium-vs-junk, then the fully-convolutional heatmap detector |

`shared.py` holds the shared palette, glyphs and the `Caption` helper so all four
clips look like one set. No LaTeX is used (Text only) — ffmpeg is required, a TeX
install is not.

## Render

`manim.cfg` pins ffmpeg and sends output to `./out/` (git-ignored).

```sh
# from this folder, using the napari-env-q python
python -m manim -ql 01_overview.py PipelineOverview      # draft 480p15
python -m manim -qh 04_cnn_heatmap.py CnnAndHeatmap      # final 1080p60
```

Quality flags: `-ql` low (fast drafts), `-qm` medium, `-qh` high, `-qk` 4K.
Rendered files land in `out/videos/<script>/<res>/<Scene>.mp4` (+ `.srt`).

## Stitch into one reel (optional)

```sh
python -m manim -qh 01_overview.py PipelineOverview
# …render all four at the same quality, then:
ffmpeg -f concat -safe 0 -i concat.txt -c copy out/reel.mp4
```

`concat.txt` lists the four mp4s in order.
