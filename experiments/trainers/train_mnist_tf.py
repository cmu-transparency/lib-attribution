import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import warnings
warnings.simplefilter('ignore')

import numpy as np
import keras
import keras.backend as K

import sys
sys.path.append('..')
from models import get_mnist_model, get_mnist_data # pylint:disable=import-error

K.set_image_data_format('channels_last')

x_tr, y_tr, x_te, y_te = get_mnist_data()
model = get_mnist_model()
model.fit(x_tr, y_tr, batch_size=512, epochs=30,
          validation_data=(x_te, y_te))
model.save_weights(os.path.join('..', 'weights', 'mnist.h5'))
