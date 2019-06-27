
import os
import keras
import numpy as np


def get_available_models():
    return ['lfw']


def get_lfw_data():
    cdir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cdir, 'data', 'lfw.npz'), 'rb') as of:
        npz = np.load(of)
        im_tr, y_tr, im_te, y_te = npz['im_tr'], npz['y_tr'], npz['im_te'], npz['y_te']

        return im_tr, y_tr, im_te, y_te


def get_lfw_model(in_shape=(64, 64, 3), out_shape=5, trained=False):
    inp = keras.layers.Input(shape=in_shape, name='features')
    out = keras.layers.Conv2D(128, (3, 3), activation='relu')(inp)
    out = keras.layers.MaxPooling2D(pool_size=(2, 2))(out)
    out = keras.layers.Conv2D(64, (3, 3), activation='relu')(out)
    out = keras.layers.MaxPooling2D(pool_size=(2, 2))(out)
    out = keras.layers.Conv2D(32, (3, 3), activation='relu')(out)
    out = keras.layers.MaxPooling2D(pool_size=(2, 2))(out)
    out = keras.layers.Conv2D(16, (3, 3), activation='relu')(out)
    out = keras.layers.MaxPooling2D(pool_size=(2, 2))(out)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(16, activation='relu')(out)
    out = keras.layers.Dense(out_shape, name='logits')(out)
    out = keras.layers.Activation('softmax', name='probs')(out)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    cdir = os.path.dirname(os.path.realpath(__file__))
    if trained and os.path.isfile(os.path.join(cdir, 'models', 'lfw.h5')):
        model.load_weights(os.path.join(cdir, 'models', 'lfw.h5'))

    return model


def get_model(name, **kwargs):
    if name == 'lfw':
        return get_lfw_model(**kwargs)
    else:
        raise ValueError('unknown model: {}'.format(name))


def get_data(name, **kwargs):
    if name == 'lfw':
        return get_lfw_data(**kwargs)
    else:
        raise ValueError('unknown model: {}'.format(name))
