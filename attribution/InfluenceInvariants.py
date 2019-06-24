import numpy as np

import keras
import keras.backend as K

from sklearn import tree

from .methods import InternalInfluence
from .invariant import Literal, Clause, Invariant


class InfluenceInvariants(object):

    def __init__(self, model, layer=None, agg_fn=None, Q=None, multiply_activation=True):
        self.model = model
        self.agg_fn = agg_fn
        self.layer = layer
        if layer is None:
            self.layer = model.layers[-2]
        elif isinstance(layer, int):
            self.layer = model.layers[layer]
        if Q is None:
            self.Q = K.argmax(model.output, axis=1)

        self._attributer = InternalInfluence(
            model, self.layer, agg_fn=agg_fn, multiply_activation=multiply_activation)
        self._is_compiled = False

    def _get_influence(self, x, batch_size=None):
        if not self._is_compiled:
            self.compile()
        if np.ndim(x) == K.ndim(self.model.input) - 1:
            x = np.expand_dims(x, axis=0)

        acts = np.sign(self._attributer.get_attributions(
            x, batch_size=batch_size)).reshape(len(x), -1)

        if batch_size is None:
            qs = self.QF(x)
        elif isinstance(batch_size, int):
            cb = 0
            qs = []
            while cb < len(x):
                b = x[cb:cb + batch_size]
                qs.append(self.QF(b))
                cb += batch_size
        else:
            raise ValueError(
                '`batch_size` must be either None or int: {}.'.format(batch_size))

        return acts, qs

    def compile(self):
        self._attributer.compile()

        f_Q = K.function([self.model.input], [self.Q])
        self.QF = lambda x: f_Q([x])[0]

        zero_input = np.zeros(shape=(1,) + K.int_shape(self.model.input)[1:])
        self._attrib_shapes = self._attributer.get_attributions(
            zero_input).shape[1:]
        self._attrib_cards = int(np.prod(self._attrib_shapes))

        feat_ranges = [(0, self._attrib_cards - 1)]
        self._feat_ranges = feat_ranges

        self._is_compiled = True

        return self

    def get_invariants(self, x, min_support=None, min_precision=1.0, batch_size=1, **kwargs):

        assert self._is_compiled

        feats, y = self._get_influence(x, batch_size=batch_size)
        clf = tree.DecisionTreeClassifier(**kwargs)
        clf = clf.fit(feats, y)

        # enumerate paths through the tree
        # adapted from the example at: https://goo.gl/8Z2XV2
        invs = []
        stack = [(0, Clause([]))]
        while len(stack) > 0:
            node, clause = stack.pop()
            if(clf.tree_.children_left[node] != clf.tree_.children_right[node]):
                unit = clf.tree_.feature[node]
                # au = K.sign(self._attributer.symbolic_attributions[:, unit])
                au = self._attributer.symbolic_attributions[:, unit]
                lit_f = Literal(self.layer, unit, K.less_equal,
                                clf.tree_.threshold[node], attribution_unit=au)
                lit_t = Literal(self.layer, unit, K.greater,
                                clf.tree_.threshold[node], attribution_unit=au)
                stack.append((clf.tree_.children_left[node],
                              clause.add_literal(lit_f, copy_self=True)))
                stack.append((clf.tree_.children_right[node],
                              clause.add_literal(lit_t, copy_self=True)))
            else:
                q = clf.tree_.value[node].argmax()
                support = clf.tree_.value[node, 0,
                                          q] / (len(np.where(y == q)[0]))
                precision = clf.tree_.value[node, 0,
                                            q] / clf.tree_.value[node].sum()
                if (min_support is None or min_support <= support) and (min_precision is None or min_precision <= precision):
                    invs.append(Invariant(
                        [clause], self._attributer.model, Q=q, support=support, precision=precision))

        return invs
