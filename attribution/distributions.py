'''
'''

import keras.backend as K


class Doi(object):
    '''
    Interface for a distribution of interest.
    '''

    def __call__(self, z):
        '''
        Takes a tensor z, which is the input to z and returns a new tensor, z',
        which has an entry for each point in the distribution of each instance.
        '''
        raise NotImplementedError('This is an abstract method.')

    @staticmethod
    def linear_interp():
        '''
        '''
        return LinearInterpDoi()

    @staticmethod
    def point():
        '''
        '''
        return PointDoi()


class LinearInterpDoi(Doi):
    '''
    '''

    def __call__(self, z):
        r = K.placeholder(name='resolution', ndim=0, shape=(), dtype='int32')
        baseline = K.placeholder(name='baseline', shape=K.int_shape(z)[1:])

        b = K.expand_dims(baseline, axis=0)

        a = K.tile(
            (1. + K.arange(r, dtype='float32')) / K.cast(r, 'float32'),
            [K.shape(z)[0]])
        for _ in range(K.ndim(z) - 1):
            a = K.expand_dims(a, axis=-1)

        # K.repeat_elements has inconsistent behavior across backends
        # For theano, it is fine to use a tensor for reps
        # For tensorflow, it is not, and repeat_elements needs a Python integer
        # The following hack for tensorflow is adapted from:
        #    https://github.com/keras-team/keras/issues/2656
        if K.backend() == 'theano':
            z_rep = K.repeat_elements(z, r, axis=0)
        elif K.backend() == 'tensorflow':
            multiples = K.variable([r]+[1 for i in range(K.ndim(z)-1)], dtype='int32')
            z_rep = K.tf.tile(z, multiples)

        #b = K.repeat_elements(b, z_rep.shape[0], axis=0)

        z_interp = b + a * (z_rep - b)

        # Set keras metadata on the resulting tensor.
        z_interp._uses_learning_phase = True
        z_interp._keras_shape = K.int_shape(z)

        # Set Doi metadata on the resulting tensor.
        z_interp._doi_parent = z
        z_interp._doi_params = {
            'resolution': r,
            'baseline': baseline}

        return z_interp


class PointDoi(Doi):
    '''
    '''

    def __call__(self, z):
        z._doi_params = {}
        return z
