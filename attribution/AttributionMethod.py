import keras

class AttributionMethod(object):
    '''
    Interface for attribution methods.

    An attribution method takes a symbolic representation of the quantity of
    interest, Q, as well as the symbolic representation of the input from the
    lower slice of the network, inpt, and compiles a function that can, given an 
    instance, compute the influence of each variable in the instance on Q.
    '''
    def __init__(self, model, layer, **kwargs):
        '''
        Parameters
        ----------
        model : keras.models.Model
            The model to calculate attributions on.
        layer : int or keras.layers.Layer or str
            The layer to calculate attributions for.
        '''
        self.model = model
        
        if isinstance(layer, int):
            self.layer = self.model.layers[layer]
        elif isinstance(layer, keras.layers.Layer):
            self.layer = layer
        elif isinstance(layer, str):
            self.layer = model.get_layer(layer)
        else:
            raise ValueError('Need to pass layer index, name, or instance.')

        self.is_compiled = False
        self.symbolic_attributions = None

  
    def compile(self, **kwargs):
        '''
        Compiles the attribution method so that `get_attributions` can be run
        efficiently.

        This should be implemented by subclasses of this class.

        Returns
        -------
        AttributionMethod
            `self`, with `self.is_compiled` set to True.
        '''
        raise NotImplementedError('This is an abstract method.')
  
    def get_attributions(self, x, **kwargs):
        '''
        TODO: docstring.
        '''
        raise NotImplementedError('This is an abstract method.')

    def _get_sym_attributions(self):
        raise NotImplementedError('This is an abstract method.')

    def get_sym_attributions(self):
        '''
        TODO: docstring.
        '''
        return (
            self.symbolic_attributions if self.symbolic_attributions else 
            self._get_sym_attributions())