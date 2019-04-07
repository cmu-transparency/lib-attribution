import numpy as np

import keras
import keras.backend as K

from .AttributionMethod import AttributionMethod

class IntegratedGradients(AttributionMethod):

    def __init__(self, model):
        AttributionMethod.__init__(self, model, 0)

class AumannShapley(AttributionMethod):

    def __init__(self, model, layer, agg_fn=K.max, Q=None, multiply_activation=True):
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
            self.attribution_units = layer_outs
            self.p_fn = lambda x: x
                        
        # Get outputs for convolutional intermediate layers
        elif K.ndim(self.layer.output) == 4:
            if self.agg_fn is None:
                n_outs = int(np.prod(K.int_shape(self.layer.output)[1:]))
                layer_outs = K.batch_flatten(self.layer.output)
                layer_grads = [inner_grad]
                self.attribution_units = K.transpose(layer_outs) #[layer_outs[:,i] for i in range(n_outs)]
                post_fn = lambda r: r[0]
                self.p_fn = lambda x: x
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
                self.attribution_units = layer_outs
            
        else:
            assert False, "Unsupported tensor shape: ndim=%d" % K.ndim(self.layer.output)

        if self.layer != self.model.layers[0]:
            feats_f = K.function([self.model.input], [self.layer.output])
            self.get_features = lambda x: np.array(feats_f([x]))[0]

        else:
            self.get_features = lambda x: x

        if hasattr(self.model, 'uses_learning_phase') and self.model.uses_learning_phase:
            grad_f = K.function([self.layer.output, K.learning_phase()], layer_grads)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp, 0])))
        else:
            grad_f = K.function([self.layer.output], layer_grads)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp])))

        self.is_compiled = True
        self.n_outs = n_outs
            
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
        elif self.agg_fn is None:
            attributions = attributions.reshape(len(attributions),-1)

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
        inner_grad = self.post_grad(K.gradients(K.sum(self.Q), self.layer.output))

        # Get outputs for flat intermediate layers
        if K.ndim(self.layer.output) == 2:
            n_outs = K.int_shape(self.layer.output)[1]
            layer_grads = inner_grad
            layer_outs = self.layer.output
            self.attribution_units = [self.layer.output[:,i] for i in range(n_outs)]
                        
        # Get outputs for convolutional intermediate layers
        # We treat each filter as an output, as in the original paper
        elif K.ndim(self.layer.output) == 4:
            if self.agg_fn is None:
                n_outs = int(np.prod(K.int_shape(self.layer.output)[1:]))
                layer_grads = K.batch_flatten(inner_grad)
                layer_outs = K.batch_flatten(self.layer.output)
                self.attribution_units = [layer_outs[:,i] for i in range(n_outs)]
            else:
                if K.image_data_format() == 'channels_first':
                    n_outs = K.int_shape(self.layer.output)[1]
                    sel_fn = lambda g: self.agg_fn(g, axis=(2,3))
                else:
                    n_outs = K.int_shape(self.layer.output)[3]
                    sel_fn = lambda g: self.agg_fn(g, axis=(1,2))
                layer_grads = sel_fn(inner_grad) # (batch, feats)
                layer_outs = sel_fn(self.layer.output) # (batch, feats)
                self.attribution_units = [layer_outs[:,i] for i in range(n_outs)]
            
        else:
            assert False, "Unsupported tensor shape: ndim=%d" % K.ndim(self.layer.output)

        if K.backend() == "theano":
            jac = K.theano.gradient.jacobian(K.sum(layer_outs, axis=0), self.model.input)
            outer_grads = [layer_grads*K.transpose(K.sum(jac, axis=(2,3,4)))]
            post_fn = lambda r: r[0]
        elif K.backend() == "tensorflow":
            outer_grads = [layer_grads[:,i] * K.sum(self.post_grad(K.gradients(K.sum(layer_outs[:,i]), self.model.input)), axis=(1,2,3))
                           for i in range(n_outs)]
            post_fn = lambda r: np.array(np.transpose(r)) #np.swapaxes(np.array(r),0,1)

        if hasattr(self.model, 'uses_learning_phase') and self.model.uses_learning_phase:
            grad_f = K.function([self.model.input, K.learning_phase()], outer_grads)
            self.dF = lambda inp: post_fn(grad_f([inp, 0]))
        else:
            grad_f = K.function([self.model.input], outer_grads)
            self.dF = lambda inp: post_fn(grad_f([inp]))

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

        attributions = np.zeros((instance.shape[0],self.n_outs)).astype(np.float32)

        for a in range(len(instance)):
            inst_a = instance[a][np.newaxis]-baseline[a]
            scale = 1./resolution
            inputs = baseline[a] + np.dot(np.repeat(inst_a, resolution, axis=0).T, np.arange(scale,1+scale,step=scale)).T
            attributions[a] = self.dF(np.expand_dims(inputs, 0)).mean(axis=0)*np.sum(inst_a)

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
            post_fn = lambda r: np.swapaxes(r,0,1)
            self.attribution_units = layer_outs
                        
        # Get outputs for convolutional intermediate layers
        elif K.ndim(self.layer.output) == 4:
            if self.agg_fn is None:
                n_outs = int(np.prod(K.int_shape(self.layer.output)[1:]))
                # K.batch_flatten seems really slow at times, so we'll save the reshape for numpy
                layer_outs = [self.layer.output]
                post_fn = lambda r: r[0].reshape((len(r[0]), -1))
                self.attribution_units = K.transpose(K.batch_flatten(self.layer.output))
            else:
                # If the aggregation function is given, treat each filter as a unit of attribution
                if K.image_data_format() == 'channels_first':
                    n_outs = K.int_shape(self.layer.output)[1]
                    sel_fn = lambda g, i: self.agg_fn(g[:,i,:,:], axis=(1,2))
                else:
                    n_outs = K.int_shape(self.layer.output)[3]
                    sel_fn = lambda g, i: self.agg_fn(g[:,:,:,i], axis=(1,2))
                layer_outs = [sel_fn(self.layer.output, i) for i in range(n_outs)]
                post_fn = lambda r: np.swapaxes(r,0,1)
                self.attribution_units = layer_outs
        else:
            assert False, "Unsupported tensor shape: ndim=%d" % K.ndim(self.layer.output)

        if hasattr(self.model, 'uses_learning_phase') and self.model.uses_learning_phase:
            grad_f = K.function([self.model.input, K.learning_phase()], layer_outs)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp, 0])))
        else:
            grad_f = K.function([self.model.input], layer_outs)
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

        attributions = self.dF(instance)

        if match_layer_shape and np.prod(K.int_shape(self.layer.output)[1:])*len(instance) == np.prod(attributions.shape):
            attributions = attributions.reshape((len(x),)+K.int_shape(self.layer.output)[1:])

        # Return in the same format as used by the caller.
        if used_batch:
            return attributions
        else:
            return attributions[0]