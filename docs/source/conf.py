# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyPheWAS"
copyright = "2025, J.T. Baker"
author = "J.T. Baker"
release = "0.3.0b1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_copybutton"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
master_doc = "index"

html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "page_width": "1200px",
    "sidebar_width": "280px",
    "github_user": "jtb324",
    "github_repo": "pyphewas-package",
    "github_type": "star",
    "github_button": True,
    "github_count": False,
}
html_title = "PyPheWAS-Package"
