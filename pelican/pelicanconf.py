#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import re
import os


SITEURL = ''
AUTHOR = u'charlesreid1'
SITENAME = u'empirical-model-building'
#SITEURL = '/sea-budgets'

PATH = 'content'
TIMEZONE = 'America/Los_Angeles'
DEFAULT_LANG = u'en'

# the theme 
THEME = 'simple-angular'



# -------------------
# ipython notebooks

HOME = os.environ.get('HOME')

PLUGIN_PATHS = [HOME+'/codes/pelican-plugins/',
                HOME+'/codes/pelican-ipynb/']

PLUGINS = ['liquid_tags','render_math','ipynb']

# ipython notebooks
MARKUP = ('md', 'ipynb')

STATIC_PATHS = ['images','notebooks']

# Don't try to turn HTML files into pages
READERS = {'html': None}

EXTRA_HEADER = open('_nb_header.html').read().decode('utf-8')




# --------------------
# Templates

'''
# template locations 
EXTRA_TEMPLATES_PATHS = ['d3',
                         'd3/budget']

# template files 
TEMPLATE_PAGES = {}

# our custom index page
TEMPLATE_PAGES['index.html'] = 'index.html'

# hello angular world
TEMPLATE_PAGES['hello.html'] = 'hello/index.html'


# budget
TEMPLATE_PAGES['budget2012.json']       = 'budget/budget2012.json'
TEMPLATE_PAGES['budget2013.json']       = 'budget/budget2013.json'
TEMPLATE_PAGES['budget2014.json']       = 'budget/budget2014.json'

TEMPLATE_PAGES['budget.html']           = 'budget/index.html'
TEMPLATE_PAGES['budget.css']            = 'budget/budget.css'
TEMPLATE_PAGES['budget_modcontrol.js']  = 'budget/budget_modcontrol.js'
TEMPLATE_PAGES['budget_sunburst.js']    = 'budget/budget_sunburst.js'
'''





# --------------8<---------------------

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
