
import numpy as np
import keras as keras
import keras.backend as K

import sys
sys.path.append('..')

from attribution.methods import InternalInfluence, AumannShapley

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

def basic_twoclass_model():

	inp = keras.layers.Input(shape=(10,))
	outp = keras.layers.Dense(10, kernel_initializer=keras.initializers.Ones(), bias_initializer=keras.initializers.Zeros())(inp)
	outp = keras.layers.Dense(2, kernel_initializer=keras.initializers.Ones(), bias_initializer=keras.initializers.Zeros())(outp)
	outp = keras.layers.Activation('softmax')(outp)
	m = keras.Model(inp, outp)
	m.compile(optimizer='sgd', loss='categorical_crossentropy')

	Wi = m.layers[1].get_weights()[0]
	Wi[:5,:5] = 0
	Wi[5:,5:] = 0
	K.set_value(m.layers[1].weights[0], Wi)
	Wf = m.layers[2].get_weights()[0]
	Wf[:5,0] = 0
	Wf[5:,1] = 0
	K.set_value(m.layers[2].weights[0], Wf)

	return m

def twoclass_model_internal_zero():
	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	# X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	attr0 = InternalInfluence(m, 1).compile().get_attributions(x0)
	attr1 = InternalInfluence(m, 1).compile().get_attributions(x1)

	return attr0[0,5:].sum() + attr1[0,:5].sum() == 0

def twoclass_model_internal_nonzero():
	m = basic_twoclass_model()
	
	# the weights are correct, half 1, half 0
	print(m.layers[1].get_weights()[0])

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 1
	# X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	attr0 = InternalInfluence(m, 1).compile().get_attributions(x0)
	attr1 = InternalInfluence(m, 1).compile().get_attributions(x1)

	# now the weights are all set to 1
	print(m.layers[1].get_weights()[0])

	return attr0[0,:5].sum() + attr1[0,5:].sum() > 0

def all_tests():
	run_test(zero_weight)
	run_test(one_weight)
	run_test(ascending_feats)
	run_test(twoclass_model_internal_zero)
	run_test(twoclass_model_internal_nonzero)
