import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import sys
sys.path.append('..')

import warnings
warnings.simplefilter('ignore')

import argparse
import time
import csv

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from collections import ChainMap

from keras.utils import to_categorical
from cleverhans.attacks import (FastGradientMethod, MadryEtAl,
                                ProjectedGradientDescent, SparseL1Descent)
from cleverhans.model import Model as CHModel

from models import get_model, get_data, get_available_models
from attribution.methods import InternalInfluence
from attribution.ActivationInvariants import ActivationInvariants
from attribution.InfluenceInvariants import InfluenceInvariants
from attribution.model_utils import replace_softmax_with_logits
from attribution.invariant_utils import merge_by_Q, inv_precision, inv_support, tally_total_stats

import logging
logging.captureWarnings(True)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('cleverhans').setLevel(logging.CRITICAL)

K.set_image_data_format('channels_last')


class KerasModel(CHModel):
    def __init__(self, model, **kwargs):
        del kwargs
        CHModel.__init__(self, 'model_b', model.output_shape[1], locals())

        self.model = model
        self.fprop(tf.placeholder(tf.float32, (128,) + model.input_shape[1:]))

    def fprop(self, x, **kwargs):
        del kwargs
        if isinstance(x, np.ndarray):
            x = K.variable(x)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            return dict([(layer.name, keras.Model(self.model.input, layer.output)(x)) for layer in self.model.layers])


def get_random_data(model, cls_size, x_shape, n_classes, nbiter=50, seed=0):
    np.random.seed(seed)
    rand_data = []
    cls_size = 256
    rand_x = np.random.uniform(size=(cls_size,) + x_shape)
    nbiter = 50
    for l in range(n_classes):
        model_ch = KerasModel(model)
        sess = K.get_session()
        pgd_params_l2 = {'eps': 5.,
                         'y_target': to_categorical(np.zeros((1,)) + l, n_classes),
                         'eps_iter': 5. / nbiter,
                         'nb_iter': nbiter,
                         'ord': 2,
                         'clip_min': 0.,
                         'clip_max': 1.}
        pgd_params_linf = {'eps': 0.1,
                           'y_target': to_categorical(np.zeros((1,)) + l, n_classes),
                           'eps_iter': 0.1 / nbiter,
                           'nb_iter': nbiter,
                           'ord': np.inf,
                           'clip_min': 0.,
                           'clip_max': 1.}
        pgd = MadryEtAl(model_ch, sess=sess)
        cur_data = []
        cur_data.append(pgd.generate_np(rand_x, **pgd_params_l2))
        cur_data.append(pgd.generate_np(rand_x, **pgd_params_linf))
        rand_data.append(np.concatenate(cur_data, axis=0))
        print('generated class', l)

    rand_data = np.concatenate(rand_data)

    return rand_data


def main():

    parser = argparse.ArgumentParser(
        description='Local linear approximations')
    parser.add_argument('--model', choices=get_available_models())
    parser.add_argument(
        '--aggfn', choices=['none', 'sum', 'max'], default='none')
    parser.add_argument('--layers', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_total', action='store_true')
    parser.add_argument('--do_average', action='store_true')
    args = parser.parse_args()

    model_nm = args.model
    agg_fn = None if args.aggfn == 'none' else K.sum if args.aggfn == 'sum' else K.max
    batch_size = args.batch_size
    layers = [int(l) for l in args.layers.split(',')]

    def np_aggfn(x, axis=None):
        return x if agg_fn is None else np.sum(x, axis=axis) if agg_fn == K.sum else np.max(x, axis=axis)

    x_tr, y_tr, x_te, y_te = get_data(model_nm)
    model = get_model(model_nm, trained=True)

    n_classes = model.output.shape[1]
    rand_data = get_random_data(
        model, 100, x_tr.shape[1:], n_classes)
    log_model = keras.Model(model.input, model.layers[-2].output)
    inflgens = [[
        InternalInfluence(log_model,
                          layer,
                          agg_fn=agg_fn,
                          Q=cl,
                          multiply_activation=False).compile()
        for cl in range(n_classes)] for layer in layers]
    print('generating invariants')
    infls = [[
        gen.get_attributions(rand_data,
                             batch_size=1,
                             resolution=100).mean(axis=0)
        for gen in gens] for gens in inflgens]
    print('compiling weights')
    weights = [np.concatenate([np.expand_dims(cinfl, axis=1)
                               for cinfl in infl], axis=1) for infl in infls]
    if agg_fn is None:
        layermods = [keras.Model(model.input, keras.layers.Flatten()(model.layers[layer].output))
                     for layer in layers]
    else:
        layermods = [keras.Model(model.input, model.layers[layer].output)
                     for layer in layers]

    def apply(x):
        return sum([np.dot(np_aggfn(lm.predict(x), axis=(1, 2)), w) for lm, w in zip(layermods, weights)])

    print('train eval')
    print(model.evaluate(x_tr, y_tr, verbose=0)[1])
    print((apply(x_tr).argmax(axis=1) == y_tr.argmax(axis=1)).mean())
    print((apply(x_tr).argmax(axis=1) == model.predict(x_tr).argmax(axis=1)).mean())

    print('test eval')
    print(model.evaluate(x_te, y_te, verbose=0)[1])
    print((apply(x_te).argmax(axis=1) == y_te.argmax(axis=1)).mean())
    print((apply(x_te).argmax(axis=1) == model.predict(x_te).argmax(axis=1)).mean())


if __name__ == '__main__':
    main()
