import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["KERAS_BACKEND"]="tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"

import sys
sys.path.append('..')

import warnings
warnings.simplefilter('ignore')

import argparse
import time

import numpy as np
import keras
import keras.backend as K

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from models import get_model, get_data, get_available_models
from attribution.ActivationInvariants import ActivationInvariants
from attribution.InfluenceInvariants import InfluenceInvariants
from attribution.model_utils import replace_softmax_with_logits
from attribution.invariant_utils import merge_by_Q

K.set_image_data_format('channels_last')

result_prefix = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'invariants_act_v_inf')

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

def plot_results(r1, r2, ylab, xlab, title, r1lab, r2lab, filename, width=0.35):
    _, ax = plt.subplots()
    inds = np.array(list(r1.keys()))
    ax.bar(inds - width/2, r1.values(), width, label=r1lab)
    ax.bar(inds + width/2, r2.values(), width, label=r2lab)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.set_xticks(list(r1.keys()))
    ax.legend()
    plt.savefig(os.path.join(result_prefix, filename))

def main():

    parser = argparse.ArgumentParser(description='Measure activation vs influence invariants')
    parser.add_argument('--model', choices=get_available_models())
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    model_nm = args.model
    batch_size = args.batch_size

    x_tr, _, x_te, _ = get_data(model_nm)
    model = get_model(model_nm, trained=True)
    model = replace_softmax_with_logits(model)    

    print('finding activation invariants')
    time0 = time.time()
    act_invgens = [ActivationInvariants(model, layers=[i], agg_fn=None).compile() for i in range(1,len(model.layers)-1)]
    act_invs = [gen.get_invariants(x_tr, batch_size=batch_size) for gen in act_invgens]
    time1 = time.time()
    print('\t[{:.2f}s]'.format(time1-time0))
    print('\n\nfinding influence invariants')
    time0 = time.time()
    inf_invgens = [InfluenceInvariants(model, layer=i, agg_fn=None).compile() for i in range(1,len(model.layers)-1)]
    inf_invs = [gen.get_invariants(x_tr, batch_size=1) for gen in inf_invgens]
    time1 = time.time()
    print('\t[{:.2f}s]'.format(time1-time0))

    if args.do_train:
        print('TRAIN')
        for layer in range(len(model.layers)-2):
            print('\tlayer {}'.format(layer), end=' ')
            sys.stdout.flush()
            time0 = time.time()
            n_per_act_tr, sup_act_tr, prec_act_tr = tally_total_invs(act_invs[layer], model, x_tr, batch_size=batch_size)
            time1 = time.time()
            print('[act: {:.2f}]'.format(time1-time0), end=' ')
            sys.stdout.flush()
            time0 = time.time()
            n_per_inf_tr, sup_inf_tr, prec_inf_tr = tally_total_invs(inf_invs[layer], model, x_tr, batch_size=batch_size)
            time1 = time.time()
            print('[inf: {:.2f}]'.format(time1-time0))
            plot_results(n_per_act_tr, n_per_inf_tr, '# invs', 'class lab.', '# invs (all classes)', 'act', 'inf', 'n_invs.png')
            plot_results(sup_act_tr, sup_inf_tr, 'support', 'class lab.', 'support (total)', 'act', 'inf', 'support_train.png')
            plot_results(prec_act_tr, prec_inf_tr, 'precision', 'class lab.', 'precision (total)', 'act', 'inf', 'precision_train.png')

    print('\nTEST')
    for layer in range(len(model.layers)-2):
        print('\tlayer {}'.format(layer), end=' ')
        sys.stdout.flush()
        time0 = time.time()
        n_per_act, sup_act_te, prec_act_te = tally_total_invs(act_invs[layer], model, x_te, batch_size=batch_size)
        time1 = time.time()
        print('[act: {:.2f}]'.format(time1-time0), end=' ')
        sys.stdout.flush()
        time0 = time.time()
        n_per_inf, sup_inf_te, prec_inf_te = tally_total_invs(inf_invs[layer], model, x_te, batch_size=batch_size)
        time1 = time.time()
        print('[inf: {:.2f}]'.format(time1-time0))
        if not args.do_train:
            plot_results(n_per_act, n_per_inf, '# invariants', 'class lab.', '# invariants (total)', 'act', 'inf', 'n_per_l{}.png'.format(layer))
        plot_results(sup_act_te, sup_inf_te, 'support', 'class lab.', 'support (total, test)', 'act', 'inf', 'support_test_l{}.png'.format(layer))
        plot_results(prec_act_te, prec_inf_te, 'precision', 'class lab.', 'precision (total, test)', 'act', 'inf', 'precision_test_l{}.png'.format(layer))
    
    

if __name__ == '__main__':
    main()