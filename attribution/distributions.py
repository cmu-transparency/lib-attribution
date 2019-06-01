'''
Docstring for the distributions module.
'''

import keras.backend as K


class Doi(object):
    '''
    Interface for a distribution of interest.

    The *distribution of interest* lets us specify the set of instances over 
    which we want our explanations to be faithful.
    '''

    def __call__(self, z):
        '''
        Takes a tensor, z, which is the input to g (in model, f = g o h), and 
        returns a new tensor, z', which has an entry for each point in the 
        distribution of interest for each instance. The distribution of interest
        is assumed be a uniform distribution over all points returned this way.

        Parameters
        ----------
        z : keras.backend.Tensor
            The tensor representing the output of the layer defining the slce.

        Returns
        -------
        keras.backend.Tensor
            A new tensor connected to `z`, which represents the distribution of
            interest.
        '''
        raise NotImplementedError('This is an abstract method.')

    @staticmethod
    def linear_interp():
        return LinearInterpDoi()

    @staticmethod
    def point():
        return PointDoi()


class LinearInterpDoi(Doi):
    '''
    A distribution of interest, which, for point, z, is the uniform distribution
    over the linear interpolation from a given baseline to z. This distribution
    of interest yields the Aumann-Shapley value.
    '''

    def __call__(self, z):
        # Make placeholders for the resolution and baseline.
        r = K.variable(10)
        baseline = K.variable(K.zeros(shape=K.int_shape(z)[1:]))

        b = K.expand_dims(baseline, axis=0)

        # Allocate the alpha term for the interpolation.
        a = K.tile(
            (1. + K.arange(r, dtype='float32')) / K.cast(r, 'float32'),
            [K.shape(z)[0]])
        for _ in range(K.ndim(z) - 1):
            a = K.expand_dims(a, axis=-1)

        # K.repeat_elements has inconsistent behavior across backends. For 
        # theano, it is fine to use a tensor for reps; for tensorflow, it is 
        # not, and repeat_elements needs a Python integer.
        # The following hack for tensorflow is adapted from:
        #    https://github.com/keras-team/keras/issues/2656
        if K.backend() == 'theano':
            z_rep = K.repeat_elements(z, r, axis=0)
        elif K.backend() == 'tensorflow':
            multiples = K.variable(
                [r]+[1 for i in range(K.ndim(z)-1)], 
                dtype='int32')
            z_rep = K.tf.tile(z, multiples)

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
    A distribution of intest where all of the probability mass is on a single 
    point.
    '''

    def __call__(self, z):
        z._doi_parent = z
        z._doi_params = {}
        return z
