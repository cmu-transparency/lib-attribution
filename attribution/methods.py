import numpy as np

import keras
import keras.backend as K

from .AttributionMethod import AttributionMethod

class IntegratedGradients(AttributionMethod):

    def __init__(self, model):
        AttributionMethod.__init__(self, model, None)

class AumannShapley(AttributionMethod):

    def __init__(self, model, layer, agg_fn=K.max, Q=None, multiply_activation=True):
        '''
        The parameters here allow us to generalize to many different methods. E.g.,

          saliency maps:         resolution=1,  multiply_activation=True
          deep taylor:           resolution=2,  multiply_activation=False
          integrated gradients:  resolution=50, multiply_activation=False
        '''
        AttributionMethod.__init__(self, model, layer)
        self.agg_fn = agg_fn
        self.multiply_activation = multiply_activation
        if Q is None:
            self.Q = self.model.output
        else:
            self.Q = Q
        if K.backend() == 'theano':
            self.post_grad = lambda x: x
        elif K.backend() == 'tensorflow':
            self.post_grad = lambda x: x[0]
        else:
            assert False, "Unsupported backend: %s" % K.backend()

    def compile(self):
        inner_grad = self.post_grad(K.gradients(K.sum(self.Q), self.layer.output))

        post_fn = lambda r: r.transpose((1,0))

        # Get outputs for flat intermediate layers
        if K.ndim(self.layer.output) == 2:
            n_outs = K.int_shape(self.layer.output)[1]
            layer_grads = [inner_grad[:,i] for i in range(n_outs)]
            layer_outs = [self.layer.output[:,i] for i in range(n_outs)]
            self.p_fn = lambda x: x
                        
        # Get outputs for convolutional intermediate layers
        elif K.ndim(self.layer.output) == 4:
            if self.agg_fn is None:
                n_outs = int(np.prod(K.int_shape(self.layer.output)[1:]))
                layer_grads = [K.batch_flatten(inner_grad)[:,i] for i in range(n_outs)]
                layer_outs = [K.batch_flatten(self.layer.output)[:,i] for i in range(n_outs)]
                self.p_fn = lambda x: x.reshape(len(x),-1)
            else:
                # If the aggregation function is given, treat each filter as a unit of attribution
                if K.image_data_format() == 'channels_first':
                    n_outs = K.int_shape(self.layer.output)[1]
                    sel_fn = lambda g, i: self.agg_fn(g[:,i,:,:], axis=(1,2))
                    p_fn = K.function([self.layer.output], [self.agg_fn(self.layer.output, axis=(2,3))])
                    self.p_fn = lambda x: p_fn([x])[0]
                else:
                    n_outs = K.int_shape(self.layer.output)[3]
                    sel_fn = lambda g, i: self.agg_fn(g[:,:,:,i], axis=(1,2))
                    p_fn = K.function([self.layer.output], [self.agg_fn(self.layer.output, axis=(1,2))])
                    self.p_fn = lambda x: p_fn([x])[0]
                layer_grads = [sel_fn(inner_grad, i) for i in range(n_outs)]
                layer_outs = [sel_fn(self.layer.output, i) for i in range(n_outs)]
            
        else:
            assert False, "Unsupported tensor shape: ndim=%d" % K.ndim(self.layer.output)

        if self.layer != self.model.layers[0]:
            feats_f = K.function([self.model.input], [self.layer.output])
            self.get_features = lambda x: np.array(feats_f([x]))[0]
        else:
            self.get_features = lambda x: x

        if self.model.uses_learning_phase:
            grad_f = K.function([self.layer.output, K.learning_phase()], layer_grads)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp, 0])))
        else:
            grad_f = K.function([self.layer.output], layer_grads)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp])))

        self.is_compiled = True
        self.n_outs = n_outs
        self.attribution_units = layer_outs
            
        return self

    def get_attributions(self, x, baseline=None, resolution=10, match_layer_shape=False):

        assert self.is_compiled, "Must compile before measuring attribution"

        if len(x.shape) == len(self.model.input_shape):
            used_batch = True
        else:
            used_batch = False
            x = np.expand_dims(x, axis=0)

        instance = self.get_features(x)

        if baseline is None:
            baseline = self.get_features(np.zeros_like(x).astype(np.float32))
        assert baseline.shape == instance.shape

        attributions = np.zeros(self.p_fn(instance).shape).astype(np.float32)

        for a in range(1, resolution + 1):
            attributions += 1.0 / resolution * self.dF(
                (instance - baseline) * a / resolution + baseline)

        if self.multiply_activation:
            attributions[:,:] *= (self.p_fn(instance) - self.p_fn(baseline))

        if match_layer_shape and np.prod(K.int_shape(self.layer.output)[1:])*len(x) == np.prod(attributions.shape):
            attributions = attributions.reshape((len(x),)+K.int_shape(self.layer.output)[1:])

        # Return in the same format as used by the caller.
        if used_batch:
            return attributions
        else:
            return attributions[0]

class Conductance(AttributionMethod):

    def __init__(self, model, layer, agg_fn=K.max, Q=None):
        AttributionMethod.__init__(self, model, layer)
        self.agg_fn = agg_fn
        if Q is None:
            self.Q = self.model.output
        else:
            self.Q = Q
        if K.backend() == 'theano':
            self.post_grad = lambda x: x
        elif K.backend() == 'tensorflow':
            self.post_grad = lambda x: x[0]
        else:
            assert False, "Unsupported backend: %s" % K.backend()

    def compile(self):
        '''
        Prepares the attributer to run on numpy inputs.
        This method must be called before get_attributions.
        
        Args:
            Q: Optional quantity of interest. The quantity of interest
                describes a property of the model's output over which
                attributions are taken. If left as None, the model's
                entire output is used.
            agg_fn: Optional aggregation function to use when self.layer
                refers to a convolutional layer. 
        
        '''        
        inner_grad = self.post_grad(K.gradients(K.sum(self.Q), self.layer.output))

        # Get outputs for flat intermediate layers
        if K.ndim(self.layer.output) == 2:
            n_outs = K.int_shape(self.layer.output)[1]
            layer_grads = [inner_grad[:,i] for i in range(n_outs)]
            layer_outs = [self.layer.output[:,i] for i in range(n_outs)]
            self.attribution_units = layer_outs
                        
        # Get outputs for convolutional intermediate layers
        # We treat each filter as an output, as in the original paper
        elif K.ndim(self.layer.output) == 4:
            if self.agg_fn is None:
                n_outs = int(np.prod(K.int_shape(self.layer.output)[1:]))
                layer_grads = K.batch_flatten(inner_grad)
                layer_outs = K.batch_flatten(self.layer.output)
                self.attribution_units = [K.batch_flatten(self.layer.output)[:,i] for i in range(n_outs)]
            else:
                if K.image_data_format() == 'channels_first':
                    n_outs = K.int_shape(self.layer.output)[1]
                    sel_fn = lambda g, i: self.agg_fn(g[:,i,:,:], axis=(1,2))
                else:
                    n_outs = K.int_shape(self.layer.output)[3]
                    sel_fn = lambda g, i: self.agg_fn(g[:,:,:,i], axis=(1,2))
                layer_grads = [sel_fn(inner_grad, i) for i in range(n_outs)]
                layer_outs = [sel_fn(self.layer.output, i) for i in range(n_outs)]
                self.attribution_units = layer_outs
            
        else:
            assert False, "Unsupported tensor shape: ndim=%d" % K.ndim(self.layer.output)

        for j in range(K.ndim(self.model.input)-1):
            for i in range(n_outs):
                layer_grads[i] = K.expand_dims(layer_grads[i], 1)

        if self.agg_fn is None and K.ndim(self.layer.output) != 2:
            outer_grads = [layer_grads[:,i] * self.post_grad(K.gradients(K.sum(layer_outs[:,i]), self.model.input))
                           for i in range(n_outs)]
        else:
            outer_grads = [layer_grads[i] * self.post_grad(K.gradients(K.sum(layer_outs[i]), self.model.input))
                           for i in range(n_outs)]

        post_fn = lambda r: np.swapaxes(r,0,1)

        gfn = K.function([self.model.input], [self.post_grad(K.gradients(K.sum(layer_outs[i]), self.model.input))
                           for i in range(n_outs)])
        self.gfn = lambda x: np.array(gfn([x]))
        ofn = K.function([self.model.input], [layer_grads[i]
                           for i in range(n_outs)])
        self.ofn = lambda x: post_fn(np.array(ofn([x])))

        if self.model.uses_learning_phase:
            grad_f = K.function([self.model.input, K.learning_phase()], outer_grads)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp, 0])))
        else:
            grad_f = K.function([self.model.input], outer_grads)
            self.grad_f = grad_f
            self.dF = lambda inp: post_fn(np.array(grad_f([inp])))

        self.is_compiled = True
        self.n_outs = n_outs

        return self

    def get_attributions(self, x, baseline=None, resolution=10, match_layer_shape=False):

        assert self.is_compiled, "Must compile before measuring attribution"

        if len(x.shape) == len(self.model.input_shape):
            used_batch = True
            instance = x
        else:
            used_batch = False
            instance = np.expand_dims(x, axis=0)

        if baseline is None:
            baseline = np.zeros_like(instance).astype(np.float32)
        assert baseline.shape == instance.shape

        attributions = np.zeros((instance.shape[0],self.n_outs) + instance.shape[1:]).astype(np.float32)

        for a in range(1, resolution + 1):
            attributions += 1.0/resolution * self.dF((instance - baseline) * a / resolution + baseline)

        for i in range(attributions.shape[1]):
            attributions[:,i] *= instance - baseline

        if np.ndim(attributions > 2):
            axes = tuple([i for i in range(np.ndim(attributions))])[2:]
            attributions = np.sum(attributions, axis=axes)

        if match_layer_shape and np.prod(K.int_shape(self.layer.output)[1:])*len(instance) == np.prod(attributions.shape):
            attributions = attributions.reshape((len(x),)+K.int_shape(self.layer.output)[1:])

        # Return in the same format as used by the caller.
        if used_batch:
            return attributions
        else:
            return attributions[0]

class Activation(AttributionMethod):

    def __init__(self, model, layer, agg_fn=K.max):
        '''

        '''
        AttributionMethod.__init__(self, model, layer)
        self.agg_fn = agg_fn

    def compile(self):

        # Get outputs for flat intermediate layers
        if K.ndim(self.layer.output) == 2:
            n_outs = K.int_shape(self.layer.output)[1]
            layer_outs = [self.layer.output[:,i] for i in range(n_outs)]
                        
        # Get outputs for convolutional intermediate layers
        elif K.ndim(self.layer.output) == 4:
            if self.agg_fn is None:
                n_outs = int(np.prod(K.int_shape(self.layer.output)[1:]))
                layer_outs = [K.batch_flatten(self.layer.output)[:,i] for i in range(n_outs)]
            else:
                # If the aggregation function is given, treat each filter as a unit of attribution
                if K.image_data_format() == 'channels_first':
                    n_outs = K.int_shape(self.layer.output)[1]
                    sel_fn = lambda g, i: self.agg_fn(g[:,i,:,:], axis=(1,2))
                else:
                    n_outs = K.int_shape(self.layer.output)[3]
                    sel_fn = lambda g, i: self.agg_fn(g[:,:,:,i], axis=(1,2))
                layer_outs = [sel_fn(self.layer.output, i) for i in range(n_outs)]
            
        else:
            assert False, "Unsupported tensor shape: ndim=%d" % K.ndim(self.layer.output)

        post_fn = lambda r: np.swapaxes(r,0,1)

        if self.model.uses_learning_phase:
            grad_f = K.function([self.model.input, K.learning_phase()], layer_outs)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp, 0])))
        else:
            grad_f = K.function([self.model.input], layer_outs)
            self.grad_f = grad_f
            self.dF = lambda inp: post_fn(np.array(grad_f([inp])))

        self.is_compiled = True
        self.n_outs = n_outs
        self.attribution_units = layer_outs
            
        return self

    def get_attributions(self, x, baseline=None, resolution=10, match_layer_shape=False):

        assert self.is_compiled, "Must compile before measuring attribution"

        if len(x.shape) == len(self.model.input_shape):
            used_batch = True
            instance = x
        else:
            used_batch = False
            instance = np.expand_dims(x, axis=0)

        attributions = self.dF(instance)

        if match_layer_shape and np.prod(K.int_shape(self.layer.output)[1:])*len(instance) == np.prod(attributions.shape):
            attributions = attributions.reshape((len(x),)+K.int_shape(self.layer.output)[1:])

        # Return in the same format as used by the caller.
        if used_batch:
            return attributions
        else:
            return attributions[0]