import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["KERAS_BACKEND"]="tensorflow"

import numpy as np
import keras as keras
import keras.backend as K

import sys
sys.path.append('..')

from attribution.methods import InternalInfluence

np.random.seed(0)

import tensorflow as tf

def zero_weight():

	inp = keras.layers.Input(shape=(10,))
	outp = keras.layers.Dense(1, kernel_initializer=keras.initializers.Zeros(), bias_initializer=keras.initializers.Zeros())(inp)
	m = keras.Model(inp, outp)
	m.compile(optimizer='sgd', loss='mse')

	x = np.ones(shape=(1,10), dtype=np.float32)
	
	attr = InternalInfluence(m, 0).compile().get_attributions(x)
	assert attr.sum() == 0, "Failed zero_weight"


if __name__ == "__main__":
	zero_weight()