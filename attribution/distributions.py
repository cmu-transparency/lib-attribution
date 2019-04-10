'''
'''

import keras.backend as K


class Doi(object):
    '''
    Interface for a distribution of interest.
    '''

    def __call__(self, z, **kwargs):
        '''
        '''
        raise NotImplementedError('This is an abstract method.')

    def get_params(self):
        return {}

    def reallocate_input(self):
        pass

    @staticmethod
    def linear_interp(baseline_shape):
        '''
        '''
        return LinearInterpDoi(baseline_shape)

    @staticmethod
    def point():
        '''
        '''
        return PointDoi()


class LinearInterpDoi(Doi):
    '''
    '''
    def __init__(self, baseline_shape, **kwargs):
        self.res = K.placeholder(shape=(1,), dtype='int32')
        self.baseline = K.placeholder(shape =(1, 1) + baseline_shape)
        self.interpolation = K.placeholder(
            shape=(1, None) + tuple(1 for _ in range(len(baseline_shape))))

    def __call__(self, z):
        return (
            (K.expand_dims(z, axis=1) - self.baseline) * self.interpolation -
            self.baseline)

    def reallocate_input(self, ):
        pass

    def get_params(self):
        return {
            'res': self.res,
            'baseline': self.baseline,
            'interpolation': self.interpolation}

class PointDoi(Doi):
    '''
    '''
    def __call__(self, z):
        return K.expand_dims(z, axis=1)
