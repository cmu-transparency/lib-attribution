import numpy as np

import keras
import keras.backend as K

from scipy import ndimage

from .VisualizationMethod import VisualizationMethod
from .AttributionMethod import AttributionMethod
from .methods import AumannShapley

class TopKWithBlur(VisualizationMethod):

    def __init__(self, attributer, k=1, percentile=95, sigma=15, alpha=0.01):
        VisualizationMethod.__init__(self, attributer)
        
        assert k > 0, "k must be a positive integer"
        self.k = k
        self.percentile = percentile
        self.sigma = sigma
        self.alpha = alpha
        if K.backend() == 'theano':
            self.post_grad = lambda x: [x]
        elif K.backend() == 'tensorflow':
            self.post_grad = lambda x: x
        else:
            assert False, "Unsupported backend: %s" % K.backend()

        self._attr_for_unit = [None for i in range(self.attributer.n_outs)]

    def visualize(self, x):
        if not self.attributer.is_compiled:
            self.attributer.compile()
        
        attribs = self.attributer.get_attributions(x)
        
        return self.visualize_np(x, attribs)
    
    def mask(self, x):
        if not self.attributer.is_compiled:
            self.attributer.compile()
        
        attribs = self.attributer.get_attributions(x)
        
        return self.mask_np(x, attribs)
    
    def visualize_np(self, x, attribs, **kwargs):
        
        return self._compute_vis(x, attribs, **kwargs)[0]
    
    def mask_np(self, x, attribs, **kwargs):
        
        return self._compute_vis(x, attribs, **kwargs)[1]
    
    def _compute_vis(self, x, attribs, **kwargs):
        if np.ndim(x) == K.ndim(self.model.input) - 1:
            x = np.expand_dims(x, axis=0)
            attribs = np.expand_dims(attribs, axis=0)
        assert np.ndim(attribs) == 2, "Unsupported attribution format: must have 2 dimensions"

        order = np.argsort(attribs, axis=1)[:,::-1][:,:self.k]

        vis = []
        masks = []
        for i in range(len(x)):

            if self.layer != self.model.layers[0]:
                Q = sum([self.attributer.attribution_units[order[i,j]] for j in range(self.k)])
                infl = AumannShapley(self.model, 0, Q=Q, agg_fn=None).compile()
                input_attrs = infl.get_attributions(x[i], match_layer_shape=True)
            else:
                input_attrs = np.zeros_like(attribs[i])
                input_attrs[order[i]] = x[i].flatten()[order[i]]
                input_attrs = input_attrs.reshape(x[i].shape)

            if K.image_data_format() == 'channels_first':
                input_attrs = input_attrs.mean(axis=0)
            else:
                input_attrs = input_attrs.mean(axis=2)
            input_attrs = np.abs(input_attrs)
            input_attrs = np.clip(input_attrs/np.percentile(input_attrs, 99), 0., 1.)
            input_attrs = ndimage.filters.gaussian_filter(input_attrs, self.sigma)

            thresh = np.percentile(input_attrs, self.percentile)
            mask = (input_attrs > thresh).astype('float32')
            mask = ndimage.filters.gaussian_filter(mask, 2)
            mask = np.clip(mask + self.alpha, 0., 1.)
            if K.image_data_format() == 'channels_first':
                mask = np.repeat(np.expand_dims(mask, 0), 3, axis=0)
            else:
                mask = np.repeat(np.expand_dims(mask, 2), 3, axis=2)

            vis.append(x[i] * mask)
            masks.append(mask)

        return np.array(vis), np.array(mask)

class UnitsWithBlur(VisualizationMethod):

    def __init__(self, attributer, units, percentile=95, sigma=15, alpha=0.1):
        VisualizationMethod.__init__(self, attributer)
        
        self.units = units
        self.percentile = percentile
        self.sigma = sigma
        self.alpha = alpha
        if K.backend() == 'theano':
            self.post_grad = lambda x: [x]
        elif K.backend() == 'tensorflow':
            self.post_grad = lambda x: x
        else:
            assert False, "Unsupported backend: %s" % K.backend()

        if not self.attributer.is_compiled:
            self.attributer.compile()

        self._attr_for_unit = [None for i in range(self.attributer.n_outs)]

    def visualize(self, x):
        
        return self.visualize_np(x, None)
    
    def mask(self, x):
        
        return self.mask_np(x, None)
    
    def visualize_np(self, x, attribs, **kwargs):
        
        return self._compute_vis(x, attribs, **kwargs)[0]
    
    def mask_np(self, x, attribs, **kwargs):
        
        return self._compute_vis(x, attribs, **kwargs)[1]
    
    def _compute_vis(self, x, attribs, **kwargs):
        if np.ndim(x) == K.ndim(self.model.input) - 1:
            x = np.expand_dims(x, axis=0)

        if self.layer != self.model.layers[0]:
            Q = sum([self.attributer.attribution_units[self.units[j]] for j in range(len(self.units))])
            infl = AumannShapley(self.model, 0, Q=Q, agg_fn=None).compile()

        vis = []
        masks = []
        for i in range(len(x)):

            if self.layer != self.model.layers[0]:
                input_attrs = infl.get_attributions(x[i], match_layer_shape=True)
            else:
                input_attrs = np.zeros_like(attribs[i])
                input_attrs[order[i]] = x[i].flatten()[order[i]]
                input_attrs = input_attrs.reshape(x[i].shape)

            if K.image_data_format() == 'channels_first':
                input_attrs = input_attrs.mean(axis=0)
            else:
                input_attrs = input_attrs.mean(axis=2)
            input_attrs = np.abs(input_attrs)
            input_attrs = np.clip(input_attrs/np.percentile(input_attrs, 99), 0., 1.)
            input_attrs = ndimage.filters.gaussian_filter(input_attrs, self.sigma)

            thresh = np.percentile(input_attrs, self.percentile)
            mask = (input_attrs > thresh).astype('float32')
            mask = ndimage.filters.gaussian_filter(mask, 2)
            mask = np.clip(mask + self.alpha, 0., 1.)
            if K.image_data_format() == 'channels_first':
                mask = np.repeat(np.expand_dims(mask, 0), 3, axis=0)
            else:
                mask = np.repeat(np.expand_dims(mask, 2), 3, axis=2)

            vis.append(x[i] * mask)
            masks.append(mask)

        return np.array(vis), np.array(mask)