# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
from datetime import datetime
from py21cmsense import __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_design",
]
if os.getenv("SPELLCHECK"):
    extensions += ("sphinxcontrib.spelling",)
    spelling_show_suggestions = True
    spelling_lang = "en_US"

autosectionlabel_prefix_document = True

autosummary_generate = True
numpydoc_show_class_members = False

source_suffix = ".rst"
master_doc = "index"
project = "21cmSense"
year = str(datetime.now().year)
author = "Jonathan Pober and Steven Murray"
copyright = "{0}, {1}".format(year, author)
version = release = __version__
templates_path = ["templates"]

pygments_style = "trac"
extlinks = {
    "issue": ("https://github.com/steven-murray/21cmSense/issues/%s", "#"),
    "pr": ("https://github.com/steven-murray/21cmSense/pull/%s", "PR #"),
}

html_theme = "furo"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
)

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "templates",
    "**.ipynb_checkpoints",
]
