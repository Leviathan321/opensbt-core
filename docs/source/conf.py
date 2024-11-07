import os
import sys
sys.path.insert(0, os.path.abspath("../../"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OpenSBT'
copyright = '2024, Lev Sorokin, Tiziano Munaro, Damir Safin'
author = 'Lev Sorokin, Tiziano Munaro, Damir Safin'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx_rtd_theme',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'sphinx.ext.mathjax' # For math support
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

html_theme = "sphinx_rtd_theme"
html_logo = '../figures/fortiss-openSBT-Logo-RGB-neg.png'
html_theme_options = {
    'logo_only': True,
    'display_version': True
}

#############
from opensbt.version import __version__

version = __version__

rst_epilog = f"""
.. |version| replace:: {version}
"""
