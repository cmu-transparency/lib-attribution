
import numpy as np
import keras as keras
import keras.backend as K

import sys
sys.path.append('..')

from attribution.methods import InternalInfluence

from test_utils import run_test

def zero_weight():

	inp = keras.layers.Input(shape=(10,))
	outp = keras.layers.Dense(1, kernel_initializer=keras.initializers.Zeros(), bias_initializer=keras.initializers.Zeros())(inp)
	m = keras.Model(inp, outp)
	m.compile(optimizer='sgd', loss='mse')

	x = np.ones(shape=(1,10), dtype=np.float32)
	
	attr = InternalInfluence(m, 0).compile().get_attributions(x)
	return attr.sum() == 0


def one_weight():

	inp = keras.layers.Input(shape=(10,))
	outp = keras.layers.Dense(1, kernel_initializer=keras.initializers.Ones(), bias_initializer=keras.initializers.Zeros())(inp)
	m = keras.Model(inp, outp)
	m.compile(optimizer='sgd', loss='mse')

	x = np.ones(shape=(1,10), dtype=np.float32)
	
	attr = InternalInfluence(m, 0).compile().get_attributions(x)
	return attr.sum() == 10

def ascending_feats():

	inp = keras.layers.Input(shape=(10,))
	outp = keras.layers.Dense(1, kernel_initializer=keras.initializers.Ones(), bias_initializer=keras.initializers.Zeros())(inp)
	m = keras.Model(inp, outp)
	m.compile(optimizer='sgd', loss='mse')

	x = np.arange(0, 10, 1, dtype=np.float32)
	
	attr = InternalInfluence(m, 0).compile().get_attributions(x)
	return (attr.argsort() == list(range(10))).all()

def all_tests():
	run_test(zero_weight)
	run_test(one_weight)
	run_test(ascending_feats)