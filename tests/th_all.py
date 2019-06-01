import os
os.environ["KERAS_BACKEND"]="theano"

import numpy as np
from all_tests import all_tests

if __name__ == "__main__":
	np.random.seed(0)
	all_tests()