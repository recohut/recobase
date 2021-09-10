
project = 'RecoHut'
copyright = '2021, Sparsh Agarwal'
author = 'Sparsh Agarwal'
release = '0.0.1'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.githubpages',
              'IPython.sphinxext.ipython_console_highlighting',
              'rst2pdf.pdfbuilder',
              'myst_parser',
]
source_suffix = ['.rst', '.ipynb']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
master_doc = 'index'
pdf_documents = [('index', u'rst2pdf', u'Sample rst2pdf doc', u'Sparsh Agarwal'),]

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))