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

K.set_image_data_format('channels_last')

def tally_total_invs(invs, model, x):

    classes = list(range(model.output.shape[1]))
    n_per_class = {cls: 0 for cls in classes}
    support = {cls: 0 for cls in classes}
    precision = {cls: 0 for cls in classes}

    for inv in invs:
        n_per_class[inv.Q] += 1
        support[inv.Q] += inv.support
        precision[inv.Q] += inv.precision*inv.support

    for cls in classes:
        if support[cls] == 0:
            continue
        precision[cls] /= float(support[cls])
    
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

    print('finding attribution invariants')
    time0 = time.time()
    act_invgens = [ActivationInvariants(model, layers=[i], agg_fn=None).compile() for i in range(1,len(model.layers)-1)]
    act_invs = [gen.get_invariants(x_tr, batch_size=batch_size) for gen in act_invgens]
    time1 = time.time()
    print('\t[{:.2f}s]'.format(time1-time0))

    print('train')
    for layer in range(len(model.layers)-2):
        n_per, sup, prec = tally_total_invs(act_invs[layer], model, x_tr)
        print('-'*10, 'layer', layer+1, '({})'.format(len(act_invs[layer])))
        print('n:', n_per)
        print('sup:', sup)
        print('prec:', prec)
    print('test')
    for layer in range(len(model.layers)-2):
        n_per, sup, prec = tally_total_invs(act_invs[layer], model, x_te)
        print('-'*10, 'layer', layer+1, '({})'.format(len(act_invs[layer])))
        print('n:', n_per)
        print('sup:', sup)
        print('prec:', prec)

    print('finding influence invariants')
    time0 = time.time()
    inf_invgens = [InfluenceInvariants(model, layer=i, agg_fn=None).compile() for i in range(1,len(model.layers)-1)]
    inf_invs = [gen.get_invariants(x_tr, batch_size=1) for gen in inf_invgens]
    time1 = time.time()
    print('\t[{:.2f}s]'.format(time1-time0))

    print('train')
    for layer in range(len(model.layers)-2):
        n_per, sup, prec = tally_total_invs(inf_invs[layer], model, x_tr)
        print('-'*10, 'layer', layer+1, '({})'.format(len(inf_invs[layer])))
        print('n:', n_per)
        print('sup:', sup)
        print('prec:', prec)

    print('test')
    for layer in range(len(model.layers)-2):
        n_per, sup, prec = tally_total_invs(inf_invs[layer], model, x_te)
        print('-'*10, 'layer', layer+1, '({})'.format(len(inf_invs[layer])))
        print('n:', n_per)
        print('sup:', sup)
        print('prec:', prec)

if __name__ == '__main__':
    main()