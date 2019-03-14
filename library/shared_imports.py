from IPython.core.display import display, HTML
import tensorflow as tf
assert tf.__version__.startswith("1.3") # the version we used
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib # plotting stuff
matplotlib.use('Agg') # for displaying plots in console without display

import library.helpers as h
from os.path import join as jp

import nltk # creating vocabularies, splitting sentences
import itertools  # creating vocabularies, splitting sentences
import os
