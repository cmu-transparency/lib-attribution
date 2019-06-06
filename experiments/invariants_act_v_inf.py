import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["KERAS_BACKEND"]="tensorflow"

import sys
sys.path.append('..')

import warnings
warnings.simplefilter('ignore')

import argparse
import time

import numpy as np
import keras
import keras.backend as K

from models import get_model, get_data, get_available_models
from attribution.ActivationInvariants import ActivationInvariants
from attribution.InfluenceInvariants import InfluenceInvariants
from attribution.model_utils import replace_softmax_with_logits
from attribution.invariant_utils import merge_by_Q

K.set_image_data_format('channels_last')

def inv_precision(inv, model, x, batch_size=None):
    cl = inv.Q
    inv_evs = inv.eval(x, batch_size=batch_size)
    inv_inds = np.where(inv_evs)[0]
    if len(inv_inds) > 0:
        x_inv = x[inv_inds]
        cl_inds = np.where(model.predict(x_inv, batch_size=batch_size).argmax(axis=1) == cl)[0]
        return float(len(cl_inds))/float(len(x_inv))
    else:
        return 1.

def inv_support(inv, model, x, batch_size=None):
    cl = inv.Q
    x_cl = x[np.where(model.predict(x, batch_size=batch_size).argmax(axis=1) == cl)[0]]
    if len(x_cl) == 0:
        return 1.
    inv_evs = inv.eval(x_cl, batch_size=batch_size)
    inv_inds = np.where(inv_evs)[0]
    return float(len(inv_inds))/float(len(x_cl))

def tally_total_invs(invs, model, x, batch_size=None):

    classes = list(range(model.output.shape[1]))
    n_per_class = {cls: 0 for cls in classes}
    support = {cls: 0 for cls in classes}
    precision = {cls: 0 for cls in classes}

    for inv in invs:
        n_per_class[inv.Q] += 1

    invs = merge_by_Q(invs)

    for inv in invs:
        support[inv.Q] += inv_support(inv, model, x, batch_size=batch_size)
        precision[inv.Q] += inv_precision(inv, model, x, batch_size=batch_size)

    return n_per_class, support, precision

def main():

    parser = argparse.ArgumentParser(description='Measure activation vs influence invariants')
    parser.add_argument('--model', choices=get_available_models())
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args(sys.argv[1:])

    model_nm = args.model
    batch_size = args.batch_size

    x_tr, y_tr, x_te, y_te = get_data(model_nm)
    model = get_model(model_nm, trained=True)
    model = replace_softmax_with_logits(model)    

    print('finding activation invariants')
    time0 = time.time()
    act_invgens = [ActivationInvariants(model, layers=[i], agg_fn=None).compile() for i in range(1,len(model.layers)-1)]
    act_invs = [gen.get_invariants(x_tr, batch_size=batch_size) for gen in act_invgens]
    time1 = time.time()
    print('\t[{:.2f}s]'.format(time1-time0))

    print('TRAIN')
    for layer in range(len(model.layers)-2):
        n_per, sup, prec = tally_total_invs(act_invs[layer], model, x_tr, batch_size=batch_size)
        print('-'*10, 'layer', layer+1, '({})'.format(len(act_invs[layer])))
        print('n:', n_per)
        print('sup:', sup)
        print('prec:', prec)
    print('\nTEST')
    for layer in range(len(model.layers)-2):
        n_per, sup, prec = tally_total_invs(act_invs[layer], model, x_te, batch_size=batch_size)
        print('-'*10, 'layer', layer+1, '({})'.format(len(act_invs[layer])))
        print('n:', n_per)
        print('sup:', sup)
        print('prec:', prec)

    print('\n\nfinding influence invariants')
    time0 = time.time()
    inf_invgens = [InfluenceInvariants(model, layer=i, agg_fn=None).compile() for i in range(1,len(model.layers)-1)]
    inf_invs = [gen.get_invariants(x_tr, batch_size=1) for gen in inf_invgens]
    time1 = time.time()
    print('\t[{:.2f}s]'.format(time1-time0))

    print('TRAIN')
    for layer in range(len(model.layers)-2):
        n_per, sup, prec = tally_total_invs(inf_invs[layer], model, x_tr, batch_size=batch_size)
        print('-'*10, 'layer', layer+1, '({})'.format(len(inf_invs[layer])))
        print('n:', n_per)
        print('sup:', sup)
        print('prec:', prec)

    print('\nTEST')
    for layer in range(len(model.layers)-2):
        n_per, sup, prec = tally_total_invs(inf_invs[layer], model, x_te, batch_size=batch_size)
        print('-'*10, 'layer', layer+1, '({})'.format(len(inf_invs[layer])))
        print('n:', n_per)
        print('sup:', sup)
        print('prec:', prec)

if __name__ == '__main__':
    main()