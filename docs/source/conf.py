# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from visualize import (
    visualize_add,
    visualize_mm,
    visualize_tiled_matrix_multiplication,
    visualize_x_tiled,
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NineToothed"
copyright = "2024, NineToothed Contributors"
author = "NineToothed Contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "NineToothed"
html_logo = "_static/ninetoothed-logo.png"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/InfiniTensor/ninetoothed",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ]
}

os.makedirs("generated", exist_ok=True)

visualize_x_tiled(4, 8, 2, 2)

visualize_add(16, 2)

visualize_tiled_matrix_multiplication(8, 8, 8, 2, 2, 2)

visualize_mm(8, 8, 8, 2, 2, 2)
