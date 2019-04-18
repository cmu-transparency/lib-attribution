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
        r = K.placeholder(ndim=0, dtype='int32')
        baseline = K.placeholder(shape=K.int_shape(z)[1:])

        b = K.expand_dims(baseline, axis=0)

        a = K.tile(
            (1. + K.arange(r, dtype='float32')) / r,
            [z.shape[0]])
        for _ in range(K.ndim(z) - 1):
            a = K.expand_dims(a, axis=-1)

        print(K.int_shape(a))

        z_rep = K.repeat_elements(z, r, axis=0)

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
