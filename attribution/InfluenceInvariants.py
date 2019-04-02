import numpy as np

import keras
import keras.backend as K

from sklearn import tree

from .methods import AumannShapley
from .invariant import Literal, Clause, Invariant

class InfluenceInvariants(object):

    def __init__(self, model, layers=None, agg_fn=K.max, Q=None):
        self.model = model
        self.agg_fn = agg_fn
        self.layers = layers
        if layers is None:
            self.layers = [model.layers[i] for i in range(1, len(self.model.layers)-1)]
        elif isinstance(layers[0], int):
            self.layers = [model.layers[i] for i in layers]
        if Q is None:
            self.Q = K.argmax(model.output, axis=1)

        self._attributers = [AumannShapley(model, layer, agg_fn=agg_fn) for layer in self.layers]
        self._is_compiled = False

    def _get_influence(self, x):
        if not self._is_compiled:
            self.compile()
        if np.ndim(x) == K.ndim(self.model.input) - 1:
            x = np.expand_dims(x, axis=0)

        acts = np.concatenate(
                [np.sign(self._attributers[i].get_attributions(x)).reshape(len(x), -1)
                    for i in range(len(self._attributers))], axis=1)

        qs = self.QF(x)

        return acts, qs

    def compile(self):
        for attrib in self._attributers:
            attrib.compile()

        f_Q = K.function([self.model.input], [self.Q])
        self.QF = lambda x: f_Q([x])[0]

        zero_input = np.zeros(shape=(1,)+K.int_shape(self.model.input)[1:])
        self._attrib_shapes = [attrib.get_attributions(zero_input).shape[1:] for attrib in self._attributers]
        self._attrib_cards = [int(np.prod(s)) for s in self._attrib_shapes]

        feat_ranges = []
        cur = 0
        for i in range(len(self.layers)):
            n = cur+self._attrib_cards[i]
            feat_ranges.append((cur,n-1))
            cur = n
        self._feat_ranges = feat_ranges

        self._is_compiled = True

        return self

    def get_invariants(self, x, min_support=None, min_precision=1.0, **kwargs):

        assert self._is_compiled

        def map_feat_to_layer(feat):
            for i in range(len(self._feat_ranges)):
                l, h = self._feat_ranges[i]
                if l <= feat and feat <= h:
                    return i, self.layers[i], feat-l
            return None

        feats, y = self._get_influence(x)
        clf = tree.DecisionTreeClassifier(**kwargs)
        clf = clf.fit(feats, y)

        # enumerate paths through the tree
        # adapted from the example at: https://goo.gl/8Z2XV2
        invs = []
        stack = [(0, Clause([]))]
        while len(stack) > 0:
            node, clause = stack.pop()
            if(clf.tree_.children_left[node] != clf.tree_.children_right[node]):
                layer, _, unit = map_feat_to_layer(clf.tree_.feature[node])
                au = self._attributers[layer].attribution_units[unit]
                lit_f = Literal(self.layers[layer], unit, 0, attribution_unit=au)
                lit_t = Literal(self.layers[layer], unit, 1, attribution_unit=au)
                stack.append((clf.tree_.children_left[node], 
                                clause.add_literal(lit_f, copy_self=True)))
                stack.append((clf.tree_.children_right[node], 
                                clause.add_literal(lit_t, copy_self=True)))
            else:
                q = clf.tree_.value[node].argmax()
                support = clf.tree_.value[node,0,q]/(len(np.where(y == q)[0]))
                precision = clf.tree_.value[node,0,q]/clf.tree_.value[node].sum()
                if (min_support is None or min_support <= support) and (min_precision is None or min_precision <= precision):
                    invs.append(Invariant([clause], self.model, Q=q, support=support, precision=precision))

        return invs