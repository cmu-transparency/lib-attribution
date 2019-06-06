import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["KERAS_BACKEND"]="tensorflow"

import warnings
warnings.simplefilter('ignore')

import numpy as np
import keras
import keras.backend as K

import sys
from ..models import get_lfw_model

K.set_image_data_format('channels_last')

from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people

# Use only classes that have at least 100 images
# There are five such classes in LFW
lfw_slice = (slice(68, 196, None), slice(61, 190, None))
faces_data = fetch_lfw_people(min_faces_per_person=100, color=True, slice_=lfw_slice)
images = faces_data.images
n_classes = faces_data.target.max()+1
x, y = faces_data.data, keras.utils.to_categorical(faces_data.target, n_classes)
images /= 255.0

# Use 3/4 for training, the rest for testing
N_tr = int(len(x)*0.75)
N_te = len(x) - N_tr
x_tr, y_tr = x[:N_tr], y[:N_tr]
x_te, y_te = x[N_tr:], y[N_tr:]
im_tr, im_te = images[:N_tr], images[N_tr:]

if not os.path.isfile('../data/lfw.npz'):
    with open('../data/data.npz', 'wb') as of:
        np.savez_compressed(of, im_tr=im_tr, y_tr=y_tr, im_te=im_te, y_te=y_te)

model = get_lfw_model()
model.fit(im_tr, y_tr, batch_size=32, epochs=40, validation_data=(im_te, y_te))
model.save_weights('../models/lfw.h5')