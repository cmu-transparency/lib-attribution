import os
os.environ["KERAS_BACKEND"]="tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from all_tests import all_tests

import tensorflow as tf
import keras

tf.logging.set_verbosity(tf.logging.FATAL)

if __name__ == "__main__":
	np.random.seed(0)
	all_tests()