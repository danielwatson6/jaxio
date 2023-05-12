# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'jaxio'
copyright = '2023, Daniel Watson'
author = 'Daniel Watson'

# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'm2r2',
    'sphinx_copybutton',
]

templates_path = ['templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_title = 'jaxio'
html_static_path = ['static']

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "toc_title": "In this page",
    "show_toc_level": 2,
}

html_context = {
    "display_github": True,
    "github_user": "danielwatson6",
    "github_repo": "jaxio", # Repo name
    "github_version": "main",
    "conf_py_path": "/source/", # Path in the checkout to the docs root
}
