import os
os.environ["KERAS_BACKEND"]="tensorflow"

import numpy as np
from invariant_tests import *

if __name__ == "__main__":
	np.random.seed(0)
	all_tests()