import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["KERAS_BACKEND"]="theano"

import numpy as np
from influence_tests import *

if __name__ == "__main__":
	np.random.seed(0)
	all_tests()