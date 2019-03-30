import keras

class AttributionMethod:
    '''
    Interface for an attribution method class.

    An attribution method takes a symbolic representation of the quantity of
    interest, Q, as well as the symbolic representation of the input from the
    lower slice of the network, inpt, and compiles a function that can, given an 
    instance, compute the influence of each variable in the instance on Q.
    '''
    def __init__(self, model, layer, **kwargs):
        self.model = model
        self.layer = None
        self.is_compiled = False
        if isinstance(layer, int):
            self.layer = self.model.layers[layer]
        if isinstance(layer, keras.layers.Layer):
            self.layer = layer
        assert self.layer is not None, "Need to pass layer index or instance"

  
    def compile(self, **kwargs):
        '''
        This should be implemented by subclasses of this class.

        - Q is the quantity of interest function. This is a tensor.
        - inpt is the input variable to the quantity of interest.

        Some models have separate behavior for training, which causes problems when
        copiling the Q function. Use has_training_mode in this case so the
        AttributionMethod can handle this case.
        '''
        raise NotImplementedError

  
    def get_attributions(self, x, **kwargs):
        '''
        Returns the influences (aka attributions) for the given instance or batch of
        instances on the quantity of interest this AttributionMethod was compiled 
        for, as a numpy array. Exactly one of instance and batch must be provided.

        The use of the single instance input is for convenience so that a single
        instance can be wrapped as a list of one instance. Batch should be used when
        this method is to be called repeatedly, as it is more efficient.

        Also returns the baseline relative to which the influences were computed, as
        a numpy array. Some attribution methods may take a baseline as a parameter,
        some may compute the baseline based on the instance, and some may not use
        a baseline at all. When a baseline is not used, the baseline returned can be
        anything.
        '''
        raise NotImplementedError