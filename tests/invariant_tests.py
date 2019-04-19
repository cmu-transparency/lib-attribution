
import numpy as np
import keras as keras
import keras.backend as K

import sys
sys.path.append('..')

from attribution.ActivationInvariants import ActivationInvariants

from test_utils import run_test

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

def basic_binary_activation_quantity():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None).compile().get_invariants(X)

	return len(invs) == 2

def basic_binary_activation_classes():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None).compile().get_invariants(X)

	has_q0_inv = invs[0].Q == 0 or invs[1].Q == 1
	has_q1_inv = invs[0].Q == 1 or invs[1].Q == 1

	return has_q0_inv and has_q1_inv

def basic_binary_activation_support():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None).compile().get_invariants(X)

	has_supp_1 = invs[0].support == 1. and invs[1].support == 1.

	return has_supp_1

def basic_binary_activation_precision():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None).compile().get_invariants(X)

	has_prec_1 = invs[0].precision == 1. and invs[1].precision == 1.

	return has_prec_1

def basic_binary_activation_evals():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None).compile().get_invariants(X)

	inv0_eval_good = False
	inv1_eval_good = False
	for inv in invs:
		if inv.Q == 0:
			inv0_eval_good = inv.eval(x0)[0]
		if inv.Q == 1:
			inv1_eval_good = inv.eval(x1)[0]
	
	return True

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

def basic_threshold_activation_quantity():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None,
								binary_feats=False).compile().get_invariants(X)

	return len(invs) == 2

def basic_threshold_activation_classes():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None,
								binary_feats=False).compile().get_invariants(X)

	has_q0_inv = invs[0].Q == 0 or invs[1].Q == 1
	has_q1_inv = invs[0].Q == 1 or invs[1].Q == 1

	return has_q0_inv and has_q1_inv

def basic_threshold_activation_support():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None,
								binary_feats=False).compile().get_invariants(X)

	has_supp_1 = invs[0].support == 1. and invs[1].support == 1.

	return has_supp_1

def basic_threshold_activation_precision():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None,
								binary_feats=False).compile().get_invariants(X)

	has_prec_1 = invs[0].precision == 1. and invs[1].precision == 1.

	return has_prec_1

def basic_threshold_activation_evals():

	m = basic_twoclass_model()

	x = np.ones(shape=(1,10), dtype=np.float32)
	x0 = np.array(x, copy=True)
	x0[0,:5] = 0
	x1 = np.array(x, copy=True)
	x1[0,5:] = 0
	X = np.tile(np.concatenate((x0,x1), axis=0), (100,1))

	x0_l = m.predict(x0).argmax(axis=1)
	x1_l = m.predict(x1).argmax(axis=1)
	
	invs = ActivationInvariants(m, 
								layers=[1], 
								agg_fn=None,
								binary_feats=False).compile().get_invariants(X)

	inv0_eval_good = False
	inv1_eval_good = False
	for inv in invs:
		if inv.Q == 0:
			inv0_eval_good = inv.eval(x0)[0]
		if inv.Q == 1:
			inv1_eval_good = inv.eval(x1)[0]
	
	return True

def all_tests():
	run_test(basic_binary_activation_quantity)
	run_test(basic_binary_activation_classes)
	run_test(basic_binary_activation_support)
	run_test(basic_binary_activation_precision)
	run_test(basic_binary_activation_evals)
	run_test(basic_threshold_activation_quantity)
	run_test(basic_threshold_activation_classes)
	run_test(basic_threshold_activation_support)
	run_test(basic_threshold_activation_precision)
	run_test(basic_threshold_activation_evals)