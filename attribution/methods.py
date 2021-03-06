'''
Docstring for the methods module.
'''

import keras.backend as K
import numpy as np
import tensorflow as tf

import keras

from keras.layers import Input

from .AttributionMethod import AttributionMethod

from .distributions import Doi
from .model_utils import top_slice, clone_model
from .quantities import Qoi


class InternalInfluence(AttributionMethod):
    '''
    Calculates attribution as the *Internal Influence* [1].

    [1] Leino et al. "Influence-directed Explanations for Deep Convolutional 
        Networks." 2018
    '''

    def __init__(
            self,
            model_in,
            layer,
            agg_fn=K.max,
            Q=None,
            D='linear_interp',
            multiply_activation=True):
        '''
        Parameters
        ----------
        model : keras.models.Model
            The model to calculate attributions on.
        layer : int or keras.layers.Layer or str
            The layer to calculate attributions for.
        agg_fn : function, optional
            A function that takes the attributions of each neuron in a feature
            map and returns a single attribution value. If `agg_fn` is None, no
            aggregation will be performed, so the attributions of each unit in
            each feature map will be returned. This parameter is not relevant
            if `layer` represents a fully-connected layer.
        Q : int or Qoi, optional
            The quantity of interest. If `Q` is None, the quantity of interest
            will default to the maximum of the model's logit output, i.e., the
            logit output for the predicted class. If `Q` is an int, the quantity
            of interest will be the logit output for the class specified by the
            given integer. Otherwise the quantity of interest will be the 
            evaluation of the given Qoi object on the top slice of `model`.
        D : str or Doi
            The distribution of interest to use when calculating the attribution
            of each point in `x`. If `distribution` is a string, it must be
            either `'linear_interp'` or `'point'`. Using `'linear_interp'`
            specifies that the distribution is the linear interpolation between
            the instance and a given *baseline* (this recovers the Aumann-
            Shapley value). Using `'point'` specifies that only the points
            themselves should be included in the distribution of interest.
        multiply_activation : bool
            If `multiply_activation` is True (default), the attribution of each
            unit will be the influence of the unit times the activation of the
            unit. Otherwise, the attribution of each unit will be simply the 
            influence of the unit.
        '''
        super(InternalInfluence, self).__init__(model_in, layer)

        # Make a clone of the model so the computation graph can be modified
        # without affecting the original model.
        self.model = clone_model(model_in)
        self.layer = self.model.get_layer(self.layer.name)

        self.agg_fn = agg_fn
        self.multiply_activation = multiply_activation

        # Get the distribution of interest.
        if isinstance(D, str):
            if D == 'point':
                self.z = Doi.point()(self.layer.output)

            elif D == 'linear_interp':
                self.z = Doi.linear_interp()(self.layer.output)

            else:
                raise ValueError(
                    "String argument `D` must be either 'linear_interp' or "
                    "'point'.")

        elif isinstance(D, Doi):
            self.z = D(self.layer.output)

        else:
            raise ValueError(
                'Argument `D` must be either a Doi object or a string.')

        # Get the top of the model.
        self.g = top_slice(
            self.model, self.layer, input_tensor=self.z, clone=False)

        if Q is None:
            # Get the maximum of the logits; this is the logit value for the
            # predicted class.

            # TODO(kleino): this needs to be the logit output.
            self.Q = K.max(self.g.output, axis=1)

        elif isinstance(Q, int):
            # Treat this as the class we would like to get the logit outputs of.
            self.Q = Qoi.for_class(Q)(self.g)

        elif isinstance(Q, Qoi):
            self.Q = Q(self.g)

        else:
            raise ValueError(
                'Argument `Q` must be either a Qoi object, an int or None.')

        # Tensorflow wraps the gradients in an array, while theano doesn't.
        if K.backend() == 'theano':
            self.post_grad = lambda x: x

        elif K.backend() == 'tensorflow':
            self.post_grad = lambda x: x[0]

        else:
            raise ValueError('Unsupported backend: {}'.format(K.backend()))

    def _get_sym_attributions(self):

        # Take gradient.
        grad = self.post_grad(K.gradients(
            K.sum(self.Q),
            self.z))
        self._sgrad = grad

        # Aggregate the gradient over the distribution of interest.
        N = K.shape(self.model.input)[0]
        D = K.int_shape(self.z)[1:]

        grad_over_doi = K.mean(K.reshape(grad, ((N, -1) + D)), axis=1)
        self.symbolic_grad = grad

        # Multiply by the activation if specified.
        if self.multiply_activation:
            if 'baseline' in self.z._doi_params:
                attribution = grad_over_doi * (
                    self.z._doi_parent -
                    K.expand_dims(self.z._doi_params['baseline'], axis=0))
            else:
                attribution = grad_over_doi * self.z._doi_parent

        else:
            attribution = grad_over_doi
        self.shape_symbolic_attributions = attribution

        # Aggregate according to aggregation function.

        # Case for flat intermediate layers.
        if K.ndim(attribution) == 2:
            # Output doesn't need to be aggregated.
            pass

        # Case for convolutional intermediate layers.
        elif K.ndim(attribution) == 4:
            if self.agg_fn is not None:
                # If the aggregation function is given, treat each feature map
                # as a unit of attribution.
                if K.image_data_format() == 'channels_first':
                    attribution = self.agg_fn(attribution, axis=(2, 3))
                else:
                    attribution = self.agg_fn(attribution, axis=(1, 2))

        else:
            raise ValueError(
                'Unsupported tensor shape: ndim={}'
                .format(K.ndim(self.layer.output)))

        self.symbolic_attributions = K.batch_flatten(attribution)

        return attribution

    def compile(self):
        x = (
            self.model.input if isinstance(self.model.input, list) else
            [self.model.input])

        doi_params = [self.z._doi_params[k] for k in self.z._doi_params]

        # If the model uses a learning phase (e.g., for dropout), we need to
        # make sure we evaluate the gradients in 'test' mode.
        learning_phase = (
            [K.learning_phase()] if
            hasattr(self.model, 'uses_learning_phase') and
            self.model.uses_learning_phase and
            K.backend() == 'theano' else
            [])

        attributions = self.get_sym_attributions()
        self.attribution_params = (x + learning_phase + doi_params)

        attribution_k_fn = K.function(self.attribution_params, [attributions])

        def attribution_fn(x, **doiparams):
            x = x if isinstance(x, list) else [x]
            lp = [0] if learning_phase else []
            doiparams = [doiparams[k] for k in doiparams]

            return attribution_k_fn(x + lp + doiparams)[0]

        self.attribution_fn = attribution_fn
        self.is_compiled = True

        return self

    def get_attributions(
            self,
            x,
            baseline=None,
            resolution=10,
            match_layer_shape=False,
            batch_size=1,
            **doi_params):
        '''
        Parameters
        ----------
        x : np.Array
            Either a single instance or a batch of instances to calculate the
            attributions for. If a single instance has shape D, a batch of
            instances should have shape N x D, where N is the number of 
            instances.
        baseline : np.Array, optional
            If the distribution of interest is given as `'linear_interp'`, 
            `baseline` specifies the baseline for the linear interpolation. When
            using `'linear_interp'`, the default baseline (used if `baseline` is
            None) is the zero vector.
        resolution: int, optional
            If the distribution of interest is given as `'linear_interp'`,
            `resolution` specifies the number of points to sample in the linear
            interpolation between the baseline and the instance. Using a
            resolution of 1 is equivalent to using `distribution='point'`.
        match_layer_shape: bool, optional
            If `match_layer_shape` is True, the resulting attributions will
            match the shape of the internal layer's output 
            (`self.layer.output_shape`). Otherwise the resulting attributions
            will be flattened to a 2-D array of attributions if `x` was given as
            a batch of instances, or a 1-D array of attributions if `x` was
            given as a single instance. If `self.layer` is a fully-connected
            layer, or if `self.agg_fn` is not None, the array of attributions 
            will match the layer shape regardless, so this parameter is 
            irrelevant.
        batch_size: int, optional
            If `batch_size` is an integer, then attributions will be computed
            by breaking `x` into batches of the specified size and concatenating
            the output afterwards. If None (default), `x` will be processed
            as-is without any batching.
        **doi_params : optional
            Additional parameters to be passed into the distribution of 
            interest.

        Returns
        -------
        np.Array
            Returns the attributions of each internal variable at layer 
            `self.layer`on the quantity of interest. If `match_layer_shape` is 
            True, and `self.agg_fn` is None, the shape of the returned array 
            will be equal to the shape of the internal layer's output 
            (`self.layer`). Otherwise, the returned array will be flattened, 
            with shape N x prod(D) if `x` was passed in as a batch of N 
            instances or prod(D) if `x` was given as a single instance.
        '''

        assert self.is_compiled, 'Must compile before measuring attribution.'

        # Figure out if we were passed an instance or a batch of instances.
        # TODO(kleino): this is not supporting models with multiple inputs.
        if len(x.shape) == len(self.model.input_shape):
            used_batch = True
        else:
            used_batch = False
            x = np.expand_dims(x, axis=0)

        # Collect the DOI parameters.
        if 'baseline' in self.z._doi_params:
            if baseline is None:
                baseline = np.zeros(K.int_shape(self.z)[1:])

            elif baseline.shape != K.int_shape(self.z)[1:]:
                raise ValueError(
                    'Shape of `baseline` must match internal layer.\n'
                    '  > shape of `baseline` was {}, but was expected to be {}.'
                    .format(baseline.shape, K.int_shape(self.z)[1:]))

            doi_params['baseline'] = baseline

        doi_params['resolution'] = np.array(resolution, dtype='int32')
        try:
            used_doi_params = {k: doi_params[k] for k in self.z._doi_params}
        except KeyError as e:
            raise ValueError(
                '`doi_params` must include parameter: {}.'.format(e))

        if batch_size is None:
            attributions = self.attribution_fn(x, **used_doi_params)
        elif isinstance(batch_size, int):
            cb = 0
            attributions = []
            while cb < len(x):
                b = x[cb:cb + batch_size]
                attributions.append(self.attribution_fn(b, **used_doi_params))
                cb += batch_size
            attributions = np.concatenate(attributions, axis=0)
        else:
            raise ValueError(
                '`batch_size` must be either None or int: {}.'.format(batch_size))

        # If we didn't specify to match the layer shape, flatten the output.
        if not match_layer_shape and self.agg_fn is None:
            attributions = attributions.reshape(len(attributions), -1)

        return attributions if used_batch else attributions[0]


class IntegratedGradients(InternalInfluence):
    '''
    [2] Sundararajan et al. "Axiomatic Attribution for Deep Networks." 2017
    '''

    def __init__(self, model):
        super(IntegratedGradients, self).__init__(model, 0, agg_fn=None)

    def get_attributions(
            self,
            x,
            baseline=None,
            resolution=10,
            match_layer_shape=False):

        return super(IntegratedGradients, self).get_attributions(
            x,
            baseline=baseline,
            resolution=resolution,
            match_layer_shape=match_layer_shape)


class SaliencyMaps(InternalInfluence):
    '''
    [3] Simonyan et al. "Deep Inside Convolutional Networks: Visualizing Image 
        Classification Models and Saliency Maps." 2013
    '''

    def __init__(self, model):
        super(SaliencyMaps, self).__init__(
            model, 0, multiply_activation=False, agg_fn=None, D='point')

    def get_attributions(self, x, match_layer_shape=False):
        return super(SaliencyMaps, self).get_attributions(
            x, match_layer_shape=match_layer_shape)


# TODO(kleino): retire this code since we're switching to `InternalInfluence`.
class AumannShapley(AttributionMethod):

    def __init__(
            self,
            model,
            layer,
            agg_fn=K.max,
            Q=None,
            multiply_activation=True):

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
        inner_grad = self.post_grad(
            K.gradients(K.sum(self.Q), self.layer.output))

        def post_fn(r): return r.transpose((1, 0))

        # Get outputs for flat intermediate layers
        if K.ndim(self.layer.output) == 2:
            n_outs = K.int_shape(self.layer.output)[1]
            layer_grads = [inner_grad[:, i] for i in range(n_outs)]
            layer_outs = [self.layer.output[:, i] for i in range(n_outs)]
            self.attribution_units = layer_outs
            self.p_fn = lambda x: x

        # Get outputs for convolutional intermediate layers
        elif K.ndim(self.layer.output) == 4:
            if self.agg_fn is None:
                n_outs = int(np.prod(K.int_shape(self.layer.output)[1:]))
                layer_outs = K.batch_flatten(self.layer.output)
                layer_grads = [inner_grad]
                self.attribution_units = K.transpose(layer_outs)

                def post_fn(r):
                    return r[0]  # pylint: disable function-redefined

                self.p_fn = lambda x: x
            else:
                # If the aggregation function is given, treat each filter as a
                # unit of attribution
                if K.image_data_format() == 'channels_first':
                    n_outs = K.int_shape(self.layer.output)[1]
                    def sel_fn(g, i): return self.agg_fn(
                        g[:, i, :, :], axis=(1, 2))
                    p_fn = K.function(
                        [self.layer.output],
                        [self.agg_fn(self.layer.output, axis=(2, 3))])
                    self.p_fn = lambda x: p_fn([x])[0]
                else:
                    n_outs = K.int_shape(self.layer.output)[3]
                    def sel_fn(g, i): return self.agg_fn(
                        g[:, :, :, i], axis=(1, 2))
                    p_fn = K.function(
                        [self.layer.output],
                        [self.agg_fn(self.layer.output, axis=(1, 2))])
                    self.p_fn = lambda x: p_fn([x])[0]
                layer_grads = [sel_fn(inner_grad, i) for i in range(n_outs)]
                layer_outs = [
                    sel_fn(self.layer.output, i) for i in range(n_outs)]
                self.attribution_units = layer_outs

        else:
            assert False, (
                'Unsupported tensor shape: ndim=%d' % K.ndim(self.layer.output))

        if self.layer != self.model.layers[0]:
            feats_f = K.function([self.model.input], [self.layer.output])
            self.get_features = lambda x: np.array(feats_f([x]))[0]

        else:
            self.get_features = lambda x: x

        if (hasattr(self.model, 'uses_learning_phase') and
                self.model.uses_learning_phase and
                K.backend() == 'theano'):

            grad_f = K.function(
                [self.layer.output, K.learning_phase()],
                layer_grads)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp, 0])))
        else:
            grad_f = K.function([self.layer.output], layer_grads)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp])))

        self.is_compiled = True
        self.n_outs = n_outs

        return self

    def get_attributions(
            self,
            x,
            baseline=None,
            resolution=10,
            match_layer_shape=False):

        assert self.is_compiled, "Must compile before measuring attribution"

        if len(x.shape) == len(self.model.input_shape):
            used_batch = True
        else:
            used_batch = False
            x = np.expand_dims(x, axis=0)

        instance = self.get_features(x)

        if baseline is None:
            baseline = self.get_features(np.zeros_like(x, dtype=np.float32))
        assert baseline.shape == instance.shape

        attributions = np.zeros(self.p_fn(instance).shape).astype(np.float32)

        for a in range(1, resolution + 1):
            attributions += 1.0 / resolution * self.dF(
                (instance - baseline) * a / resolution + baseline)

        if self.multiply_activation:
            attributions[:, :] *= (self.p_fn(instance) - self.p_fn(baseline))

        if (match_layer_shape and
                np.prod(K.int_shape(self.layer.output)[1:]) * len(x) ==
                np.prod(attributions.shape)):

            attributions = attributions.reshape(
                (len(x),) + K.int_shape(self.layer.output)[1:])
        elif self.agg_fn is None:
            attributions = attributions.reshape(len(attributions), -1)

        # Return in the same format as used by the caller.
        if used_batch:
            return attributions
        else:
            return attributions[0]


class Conductance(AttributionMethod):
    '''
    Paper: 
    '''

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
        inner_grad = self.post_grad(
            K.gradients(K.sum(self.Q), self.layer.output))

        # Get outputs for flat intermediate layers
        if K.ndim(self.layer.output) == 2:
            n_outs = K.int_shape(self.layer.output)[1]
            layer_grads = inner_grad
            layer_outs = self.layer.output
            self.attribution_units = [self.layer.output[:, i]
                                      for i in range(n_outs)]

        # Get outputs for convolutional intermediate layers
        # We treat each filter as an output, as in the original paper
        elif K.ndim(self.layer.output) == 4:
            if self.agg_fn is None:
                n_outs = int(np.prod(K.int_shape(self.layer.output)[1:]))
                layer_grads = K.batch_flatten(inner_grad)
                layer_outs = K.batch_flatten(self.layer.output)
                self.attribution_units = [layer_outs[:, i]
                                          for i in range(n_outs)]
            else:
                if K.image_data_format() == 'channels_first':
                    n_outs = K.int_shape(self.layer.output)[1]
                    def sel_fn(g): return self.agg_fn(g, axis=(2, 3))
                else:
                    n_outs = K.int_shape(self.layer.output)[3]
                    def sel_fn(g): return self.agg_fn(g, axis=(1, 2))
                layer_grads = sel_fn(inner_grad)  # (batch, feats)
                layer_outs = sel_fn(self.layer.output)  # (batch, feats)
                self.attribution_units = [layer_outs[:, i]
                                          for i in range(n_outs)]

        else:
            assert False, "Unsupported tensor shape: ndim=%d" % K.ndim(
                self.layer.output)

        if K.backend() == "theano":
            jac = K.theano.gradient.jacobian(
                K.sum(layer_outs, axis=0), self.model.input)
            outer_grads = [layer_grads *
                           K.transpose(K.sum(jac, axis=(2, 3, 4)))]

            def post_fn(r): return r[0]
        elif K.backend() == "tensorflow":
            outer_grads = [layer_grads[:, i] * K.sum(self.post_grad(K.gradients(K.sum(layer_outs[:, i]), self.model.input)), axis=(1, 2, 3))
                           for i in range(n_outs)]

            # np.swapaxes(np.array(r),0,1)
            def post_fn(r): return np.array(np.transpose(r))

        if hasattr(self.model, 'uses_learning_phase') and self.model.uses_learning_phase and K.backend() == 'theano':
            grad_f = K.function(
                [self.model.input, K.learning_phase()], outer_grads)
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
            baseline = np.zeros_like(instance, dtype=np.float32)
        assert baseline.shape == instance.shape

        attributions = np.zeros(
            (instance.shape[0], self.n_outs)).astype(np.float32)

        for a in range(len(instance)):
            inst_a = instance[a][np.newaxis] - baseline[a]
            scale = 1. / resolution
            inputs = baseline[a] + np.dot(np.repeat(inst_a, resolution,
                                                    axis=0).T, np.arange(scale, 1 + scale, step=scale)).T
            attributions[a] = self.dF(np.expand_dims(
                inputs, 0)).mean(axis=0) * np.sum(inst_a)

        if match_layer_shape and np.prod(K.int_shape(self.layer.output)[1:]) * len(instance) == np.prod(attributions.shape):
            attributions = attributions.reshape(
                (len(x),) + K.int_shape(self.layer.output)[1:])

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

        # Get outputs for flat intermediate layers.
        if K.ndim(self.layer.output) == 2:
            n_outs = K.int_shape(self.layer.output)[1]
            layer_outs = [self.layer.output[:, i] for i in range(n_outs)]
            def post_fn(r): return np.swapaxes(r, 0, 1)

            self.attribution_units = layer_outs

        # Get outputs for convolutional intermediate layers.
        elif K.ndim(self.layer.output) == 4:
            if self.agg_fn is None:
                n_outs = int(np.prod(K.int_shape(self.layer.output)[1:]))
                # K.batch_flatten seems really slow at times, so we'll save the
                # reshape for numpy.
                layer_outs = [self.layer.output]
                def post_fn(r): return r[0].reshape((len(r[0]), -1))

                self.attribution_units = K.transpose(
                    K.batch_flatten(self.layer.output))
            else:
                # If the aggregation function is given, treat each filter as a
                # unit of attribution.
                if K.image_data_format() == 'channels_first':
                    n_outs = K.int_shape(self.layer.output)[1]
                    def sel_fn(g, i): return self.agg_fn(
                        g[:, i, :, :], axis=(1, 2))
                else:
                    n_outs = K.int_shape(self.layer.output)[3]
                    def sel_fn(g, i): return self.agg_fn(
                        g[:, :, :, i], axis=(1, 2))
                layer_outs = [
                    sel_fn(self.layer.output, i) for i in range(n_outs)]

                def post_fn(r): return np.swapaxes(r, 0, 1)

                self.attribution_units = layer_outs
        else:
            raise ValueError(
                'Unsupported tensor shape: ndim={}'
                .format(K.ndim(self.layer.output)))

        if (hasattr(self.model, 'uses_learning_phase') and
                self.model.uses_learning_phase and
                K.backend() == 'theano'):

            grad_f = K.function(
                [self.model.input, K.learning_phase()],
                layer_outs)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp, 0])))

        else:
            grad_f = K.function([self.model.input], layer_outs)
            self.dF = lambda inp: post_fn(np.array(grad_f([inp])))

        self.is_compiled = True
        self.n_outs = n_outs

        return self

    def get_attributions(
            self,
            x,
            baseline=None,
            resolution=10,
            batch_size=None,
            match_layer_shape=False):

        assert self.is_compiled, 'Must compile before measuring attribution.'

        if len(x.shape) == len(self.model.input_shape):
            used_batch = True
            instance = x
        else:
            used_batch = False
            instance = np.expand_dims(x, axis=0)

        if batch_size is None:
            attributions = self.dF(instance)
        elif isinstance(batch_size, int):
            cb = 0
            attributions = []
            while cb < len(x):
                b = x[cb:cb + batch_size]
                attributions.append(self.dF(b))
                cb += batch_size
            attributions = np.concatenate(attributions, axis=0)
        else:
            raise ValueError(
                '`batch_size` must be either None or int: {}.'.format(batch_size))

        if (match_layer_shape and
                np.prod(K.int_shape(self.layer.output)[1:]) * len(instance) ==
                np.prod(attributions.shape)):

            attributions = attributions.reshape(
                (len(x),) + K.int_shape(self.layer.output)[1:])

        # Return in the same format as used by the caller.
        return attributions if used_batch else attributions[0]
