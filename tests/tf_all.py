import os
os.environ["KERAS_BACKEND"]="tensorflow"

import numpy as np
from all_tests import all_tests

import tensorflow as tf
import keras

if __name__ == "__main__":
	np.random.seed(0)
	all_tests()