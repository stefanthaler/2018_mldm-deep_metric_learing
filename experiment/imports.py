import tensorflow as tf
assert tf.__version__.startswith("1.4") # the version we used
import numpy as np
import logging

import os
from os.path import join as jp
import library.helpers as h
import library.tensorflow_helpers as tfh
import time
from library.vocabulary import *
from tensorflow.contrib.tensorboard.plugins import projector # for visualizing embeddings
import re

import matplotlib # plotting stuff
matplotlib.use('Agg') # for displaying plots in console without display
