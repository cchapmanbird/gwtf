# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pygwtf"
copyright = "2026, Christian Chapman-Bird, Diganta Bandopadhyay"
author = "Christian Chapman-Bird, Diganta Bandopadhyay"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autoapi.extension", "nbsphinx"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoapi_type = "python"
autoapi_dirs = ["../src/pygwtf/"]
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "imported-members",
    "show-inheritance",
    "show-module-summary",
    "undoc-members",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "pygwtf"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/cchapmanbird/gwtf",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
