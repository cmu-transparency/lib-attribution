import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
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
import keras
import keras.backend as K

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from collections import ChainMap

from models import get_model, get_data, get_available_models
from attribution.ActivationInvariants import ActivationInvariants
from attribution.InfluenceInvariants import InfluenceInvariants
from attribution.model_utils import replace_softmax_with_logits
from attribution.invariant_utils import merge_by_Q, inv_precision, inv_support, tally_total_stats

K.set_image_data_format('channels_last')

result_prefix = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'results', 'invariants_act_v_inf')


def plot_results(r1, r2, ylab, xlab, title, r1lab, r2lab, filename, width=0.35):
    _, ax = plt.subplots()
    inds = np.array(list(r1.keys()))
    ax.bar(inds - width / 2, r1.values(), width, label=r1lab)
    ax.bar(inds + width / 2, r2.values(), width, label=r2lab)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.set_xticks(list(r1.keys()))
    ax.legend()
    plt.savefig(os.path.join(result_prefix, filename + '.png'))


def invariants_csv(invs_by_layer, filename):
    features = ['layer', 'Q', 'support', 'precision']
    csvfile = open(os.path.join(
        result_prefix, filename + '.csv'), 'w', newline='')
    writer = csv.DictWriter(
        csvfile, fieldnames=features, quoting=csv.QUOTE_NONE)

    for layer in range(len(invs_by_layer)):
        invs = invs_by_layer[layer]
        for inv in invs:
            csvrow = {'layer': layer,
                      'Q': inv.Q,
                      'support': '{:.4}'.format(inv.support),
                      'precision': '{:.4}'.format(inv.precision)}
            writer.writerow(csvrow)
            csvfile.flush()

    csvfile.close()


def do_total(x, model, act_invs, inf_invs, batch_size):
    features = ['layer', 'class_lab', 'n_invs_act', 'n_invs_inf',
                'support_act', 'support_inf', 'precision_act', 'precision_inf']
    csvfile = open(os.path.join(
        result_prefix, 'stats_total_test.csv'), 'w', newline='')
    writer = csv.DictWriter(
        csvfile, fieldnames=features, quoting=csv.QUOTE_NONE)
    writer.writeheader()
    for layer in range(len(model.layers) - 2):
        print('\tlayer {}'.format(layer), end=' ')
        sys.stdout.flush()
        time0 = time.time()
        n_per_act, sup_act_te, prec_act_te = tally_total_stats(
            act_invs[layer], model, x, batch_size=batch_size)
        time1 = time.time()
        print('[act: {:.2f}]'.format(time1 - time0), end=' ')
        sys.stdout.flush()
        time0 = time.time()
        n_per_inf, sup_inf_te, prec_inf_te = tally_total_stats(
            inf_invs[layer], model, x, batch_size=batch_size)
        time1 = time.time()
        print('[inf: {:.2f}]'.format(time1 - time0))
        plot_results(n_per_act, n_per_inf, '# invariants', 'class lab.',
                     '# invariants (total)', 'act', 'inf', 'n_per_l{}'.format(layer))
        plot_results(sup_act_te, sup_inf_te, 'support', 'class lab.',
                     'support (total, test)', 'act', 'inf', 'support_test_l{}'.format(layer))
        plot_results(prec_act_te, prec_inf_te, 'precision', 'class lab.',
                     'precision (total, test)', 'act', 'inf', 'precision_test_l{}'.format(layer))

        for cl in n_per_act.keys():
            csvrow = ({'layer': layer,
                       'class_lab': cl,
                       'n_invs_act': n_per_act[cl],
                       'n_invs_inf': n_per_inf[cl],
                       'support_act': '{:.4}'.format(sup_act_te[cl]),
                       'support_inf': '{:.4}'.format(sup_inf_te[cl]),
                       'precision_act': '{:.4}'.format(prec_act_te[cl]),
                       'precision_inf': '{:.4}'.format(prec_inf_te[cl])})
            writer.writerow(csvrow)
        csvfile.flush()
    csvfile.close()


def main():

    parser = argparse.ArgumentParser(
        description='Measure activation vs influence invariants')
    parser.add_argument('--model', choices=get_available_models())
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_total', action='store_true')
    parser.add_argument('--do_average', action='store_true')
    args = parser.parse_args()

    model_nm = args.model
    batch_size = args.batch_size

    x_tr, _, x_te, _ = get_data(model_nm)
    model = get_model(model_nm, trained=True)
    model = replace_softmax_with_logits(model)

    print('finding activation invariants')
    time0 = time.time()
    act_invgens = [ActivationInvariants(model, layers=[i], agg_fn=None).compile(
    ) for i in range(1, len(model.layers) - 1)]
    act_invs = [gen.get_invariants(x_tr, batch_size=batch_size)
                for gen in act_invgens]
    time1 = time.time()
    print('\t[{:.2f}s]'.format(time1 - time0))
    invariants_csv(act_invs, 'invariants_act')
    print('\n\nfinding influence invariants')
    time0 = time.time()
    inf_invgens = [InfluenceInvariants(model, layer=i, agg_fn=None, multiply_activation=False).compile(
    ) for i in range(1, len(model.layers) - 1)]
    inf_invs = [gen.get_invariants(x_tr, batch_size=1) for gen in inf_invgens]
    time1 = time.time()
    print('\t[{:.2f}s]'.format(time1 - time0))
    invariants_csv(inf_invs, 'invariants_inf')

    if args.do_total:
        do_total(x_te, model, act_invs, inf_invs, batch_size)


if __name__ == '__main__':
    main()
