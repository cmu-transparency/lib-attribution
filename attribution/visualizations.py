import numpy as np

import keras
import keras.backend as K

from scipy import ndimage

from .VisualizationMethod import VisualizationMethod
from .AttributionMethod import AttributionMethod
from .methods import AumannShapley

class TopKWithBlur(VisualizationMethod):

	def __init__(self, attributer, k=1, percentile=95, sigma=3, alpha=0.01):
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

	def _get_attr_for_unit(self, i, x):

		if self._attr_for_unit[i] is None:
			self._attr_for_unit[i] = AumannShapley(self.model, 0, Q=self.attributer.attribution_units[i], agg_fn=None).compile()

		return self._attr_for_unit[i].get_attributions(x, match_layer_shape=True)

	def visualize(self, x):
		if not self.attributer.is_compiled:
			self.attributer.compile()
		
		attribs = self.attributer.get_attributions(x)
		
		return self.visualize_np(x, attribs)
	
	def visualize_np(self, x, attribs, **kwargs):
		if np.ndim(x) == K.ndim(self.model.input) - 1:
			x = np.expand_dims(x, axis=0)
			attribs = np.expand_dims(attribs, axis=0)
		assert np.ndim(attribs) == 2, "Unsupported attribution format: must have 2 dimensions"

		order = np.argsort(attribs, axis=1)[:,::-1][:,:self.k]

		vis = []
		for i in range(len(x)):

			if self.layer != self.model.layers[0]:
				input_attrs = np.sum([
					np.expand_dims(self._get_attr_for_unit(order[i,j], x[i]), 0) 
						for j in range(len(order[i]))], axis=0)[0]
			else:
				input_attrs = np.zeros_like(attribs[i])
				input_attrs[order[i]] = x[i].flatten()[order[i]]
				input_attrs = input_attrs.reshape(x[i].shape)

			in_min = input_attrs.min()
			if in_min < 0:
				input_attrs += in_min
			in_max = input_attrs.max() + 1.E-8
			input_attrs /= in_max

			thresh = np.percentile(input_attrs, self.percentile)
			mask = (input_attrs > thresh).astype('float32')
			mask = ndimage.filters.gaussian_filter(mask, self.sigma)
			mask = np.clip(mask + self.alpha, 0., 1.)

			vis.append(x[i] * mask)

		return np.array(vis)