import argparse
import csv
import glob
import logging
import os
import re
import sys
import time
import warnings
from os import path

import cv2
import keras
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import tensorflow as tf
from keras import backend as K
from keras import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import Progbar
from matplotlib.image import imread

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append('..')
warnings.simplefilter('ignore')
logging.captureWarnings(True)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
K.set_image_data_format('channels_last')

MODEL_DIR = '/longterm/shared_models/glasses-attack/face-rec-attacks/'
sys.path.append(MODEL_DIR)

from FaceRecognitionNets import VGGNet  # pylint: disable=import-error
from attribution.methods import AumannShapley
from attribution.visualizations import TopKWithBlur, UnitsWithBlur


def normalize_images(ims):
    ims_n = np.array(ims, copy=True)
    ims_n[:, :, :, 0] = ims[:, :, :, 0] - VGGNet.average_image[0]
    ims_n[:, :, :, 1] = ims[:, :, :, 1] - VGGNet.average_image[1]
    ims_n[:, :, :, 2] = ims[:, :, :, 2] - VGGNet.average_image[2]
    return ims_n


def load_all_benign(model_dir):
    imdir = os.path.join(MODEL_DIR, 'images', 'benign')
    imname = 'vgg10-*-*.png'
    files = glob.glob(os.path.join(imdir, imname))
    rec = re.compile(r'.*/vgg10-([0-9]+)-.*.png')
    labels = list(set([int(rec.match(fn).group(1)) for fn in files]))
    ims = {label: [] for label in labels}
    for file in files:
        label = int(rec.match(file).group(1))
        im = imread(file) * 255.
        im = im[:, :, :3]
        im = cv2.resize(im, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)  # pylint: disable=no-member
        ims[label].append(im[np.newaxis])
    for label in labels:
        ims[label] = np.concatenate(ims[label], axis=0)
    return ims


def load_all_impersonation(model_dir):
    imdir = os.path.join(MODEL_DIR, 'images', 'impersonation')
    imname = 'vgg10-*-*-*.png'
    files = glob.glob(os.path.join(imdir, imname))
    rec = re.compile(r'.*/vgg10-([0-9]+)-([0-9]+)-.*.png')
    labels = list(set([int(rec.match(fn).group(1)) for fn in files]))
    ims = {label: [] for label in labels}
    targets = {label: [] for label in labels}
    for file in files:
        label = int(rec.match(file).group(1))
        target = int(rec.match(file).group(2))
        im = imread(file) * 255.
        ims[label].append(im[:, :, :3][np.newaxis])
        targets[label].append(target)
    for label in labels:
        ims[label] = np.concatenate(ims[label], axis=0)
    return ims, targets


def get_smooth_mask(im_t_n, asa, vistop, nsmooth=100):
    im_t_n_s = np.random.normal(loc=0,
                                scale=np.sqrt(im_t_n.var()),
                                size=(nsmooth,) + im_t_n.shape)[:] + im_t_n
    attrs = asa.get_attributions(im_t_n_s).mean(axis=0)
    mask = vistop.mask_np(im_t_n, attrs)
    return mask


def main():

    parser = argparse.ArgumentParser(
        description='Local linear approximations')
    parser.add_argument(
        '--aggfn', choices=['none', 'sum', 'max'], default='max')
    parser.add_argument('--layer', type=int, default='16')
    parser.add_argument('--nsmooth', type=int, default=50)
    parser.add_argument('--nunit', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=1.)
    parser.add_argument('--percentile', type=int, default=95)
    parser.add_argument('--multiply_act', action='store_false')
    parser.add_argument('--imp_labels', type=str, default='7,8,9')
    args = parser.parse_args()

    layer = args.layer
    nsmooth = args.nsmooth
    nunit = args.nunit
    sigma = args.sigma
    percentile = args.percentile
    multiply_act = args.multiply_act
    agg_fn = None if args.aggfn == 'none' else K.sum if args.aggfn == 'sum' else K.max
    imp_labels = [int(l) for l in args.imp_labels.split(',')]

    print('using config:\n\tlayer:\t\t{}\n\tnsm:\t\t{}\n\tnunit:\t\t{}\n\tsigma:\t\t{}\n\tpctile:\t\t{}\n\tmulact:\t\t{}\n\taggfn:\t\t{}\n\timps:\t\t{}'.format(
        layer, nsmooth, nunit, sigma, percentile, multiply_act, args.aggfn, args.imp_labels))

    with open('%s/aux-data/names-10classes.txt' % MODEL_DIR, mode='r') as fin:
        names = fin.read().split('\n')

    vgg_model = VGGNet('%s/weights/vgg10-recognition-nn-raw-weights.mat' % MODEL_DIR)
    model = vgg_model.model
    log_model = Model(model.input, model.layers[-2].output)

    n_classes = model.output_shape[1]
    benign_ims = load_all_benign(MODEL_DIR)
    benign_ims_n = {label: normalize_images(benign_ims[label]) for label in benign_ims.keys()}
    all_be_x = np.concatenate([benign_ims[label] for label in benign_ims.keys()], axis=0)
    all_be_x_n = normalize_images(all_be_x)
    all_be_y = np.concatenate([to_categorical(np.repeat([label], len(benign_ims[label])),
                                              n_classes) for label in benign_ims.keys()], axis=0)
    imp_ims, imp_targets = load_all_impersonation(MODEL_DIR)
    imp_ims_n = {label: normalize_images(imp_ims[label]) for label in imp_ims.keys()}
    all_imp_x = np.concatenate([imp_ims[label] for label in imp_ims.keys()], axis=0)
    all_imp_x_n = normalize_images(all_imp_x)
    all_imp_y = np.concatenate([to_categorical(np.repeat([label], len(imp_ims[label])),
                                               n_classes) for label in imp_ims.keys()], axis=0)
    all_imp_t = np.concatenate([
        np.concatenate([
            to_categorical(t, n_classes)[np.newaxis]
            for t in imp_targets[label]], axis=0)
        for label in imp_ims.keys()], axis=0)

    for l in imp_labels:
        label_avg = 0
        n_ims = len(imp_ims_n[l])

        asa = AumannShapley(log_model,
                            layer,
                            agg_fn=agg_fn,
                            Q=log_model.output[:, imp_targets[l][0]],
                            multiply_activation=multiply_act).compile()
        vistop = TopKWithBlur(asa, k=nunit, sigma=sigma, percentile=percentile, alpha=0.)
        if imp_targets[l][0] in benign_ims.keys() and False:
            mean_vic = benign_ims_n[imp_targets[l][0]].mean(axis=0)
        else:
            mean_vic = all_be_x_n.mean(axis=0)

        print('measuring impersonator: {}'.format(names[l]))
        pb = Progbar(n_ims)
        for i in range(n_ims):
            im_t_n = imp_ims_n[l][i]
            mask = get_smooth_mask(im_t_n, asa, vistop, nsmooth)
            im_r_n = im_t_n * (1. - mask) + mean_vic * mask
            p_mask = model.predict(im_t_n[np.newaxis] * (1. - mask))
            p_repl = model.predict(im_r_n[np.newaxis])
            pb.add(1, [('succh', 1. if p_mask.argmax() == l else 0.), ('succs', 1. if p_mask.argmax()
                                                                       != imp_targets[l][i] else 0.), ('succr', 1. if p_repl.argmax() == l else 0.)])
            label_avg += 1. / float(n_ims) if p_mask.argmax() == l else 0.
        print('\n{} success rate: {:.2}'.format(names[l], label_avg))


if __name__ == '__main__':
    main()
