"""Sphinx configuration for the LimonCELLo wiki.

Docs are authored in Markdown and parsed by MyST, so the same ``.md`` files render
on GitHub *and* build into this site. Build with::

    cd docs
    python -m sphinx -b html . _build/html
    # or:  make html   (make.bat on Windows)
"""

project = "LimonCELLo"
author = "daggermaster3000"
copyright = "2026, LimonCELLo contributors"

# -- General ------------------------------------------------------------------
extensions = [
    "myst_parser",            # Markdown support
    "sphinx.ext.autosectionlabel",
]

# Markdown is the source format; the landing page is Home.md.
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "Home"

# MyST niceties: ::: fences, definition lists, task lists, ${} substitutions.
myst_enable_extensions = ["colon_fence", "deflist", "tasklist", "substitution"]
myst_heading_anchors = 3          # auto-anchors for h1–h3 (cross-page links)

# Prefix autosectionlabel refs with the document so duplicate headings don't clash.
autosectionlabel_prefix_document = True

# Not part of the built docs: the build dir, the screenshot drop-guide, raw assets.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "images/README.md"]

# Missing screenshots (images/*.png you haven't added yet) are warnings, not
# errors — the build still succeeds.
suppress_warnings = ["image.not_readable"]

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "LimonCELLo"
html_static_path = ["_static"]     # optional custom CSS/JS lives here
# Screenshots referenced as images/*.png and assets/*.png are copied as-is.

# Copied verbatim to the output root: a redirect ``index.html`` → ``Home.html``
# (so the site root works on GitHub Pages) and ``.nojekyll`` (so ``_static`` is
# served, not eaten by Jekyll).
html_extra_path = ["_extra"]

html_theme_options = {
    "source_repository": "https://github.com/daggermaster3000/LimonCELLo/",
    "source_branch": "main",
    "source_directory": "docs/",
}
