"""
This script converts the tracklists of the playlists into sequences
of word indices into the embedding matrix from the 100-dimensional
pre-trained GloVe vectors found here:
    http://nlp.stanford.edu/projects/glove/
"""


import re
import sys

import tensorflow as tf
import numpy as np


if __name__ == '__main__':

