import numpy as np

import keras
import keras.backend as K
import copy

class Literal(object):

    def __init__(self, layer, unit, op, value, attribution_unit=None):
        self.layer = layer
        self.unit = unit
        self.value = value
        self.attribution_unit = attribution_unit
        self.op = op

    def get_tensor(self, attribution_unit=None):
        u_t = attribution_unit if attribution_unit is not None else self.attribution_unit
        assert u_t is not None, "Need to pass attribution unit tensor either directly or in constructor"

        zeros = K.zeros_like(u_t)
        ones = K.ones_like(u_t)
        vals = zeros + self.value
        u_bt = K.expand_dims(K.switch(self.op(u_t, vals), ones, zeros), 1)

        return u_bt

    def __deepcopy__(self, memo):
        return copy.copy(self)

    def __str__(self):
        op = '?'
        if self.op == K.equal:
            op = '='
        if self.op == K.not_equal:
            op = '!='
        if self.op == K.less:
            op = '<'
        if self.op == K.less_equal:
            op = '<='
        if self.op == K.greater:
            op = '>'
        if self.op == K.greater_equal:
            op = '>='

        return "{}[{}] {} {}".format(self.layer.name, self.unit, op, self.value)

class Clause(object):
    '''
    Conjunctive clause
    '''
    def __init__(self, literals):
        self.literals = literals

    def add_literal(self, lit, copy_self=False):
        t = copy.deepcopy(self) if copy_self else self
        t.literals.append(lit)
        return t

    def get_tensor(self, attributers=None):
        if attributers is not None:
            clause = [lit.get_tensor(attribution_unit=attributers[lit.layer].attribution_units[lit.unit]) for lit in self.literals]
        else:
            clause = [lit.get_tensor() for lit in self.literals]

        return K.expand_dims(K.all(K.concatenate(clause, axis=1), axis=1), axis=1)

    def get_executable(self, model, attributers=None):
        t = self.get_tensor(attributers=attributers)
        exe = K.function([model.input], [t])

        return lambda x: exe([x])[0]

    def __str__(self):
        return " &\n ".join([str(literal) for literal in self.literals])

class Invariant(object):

    def __init__(self, clauses, model, Q=None, support=None, precision=None):
        self.clauses = clauses
        self.Q = Q
        self.support = support
        self.precision = precision
        self.model = model
        self._T = None
        self._exe = None

    def add_clause(self, clause, copy_self=False):
        t = copy.deepcopy(self) if copy_self else self
        t.clauses.append(clause)

        self._T = None
        self._exe = None

        return t

    def get_tensor(self, attributers=None):
        if self._T is None:
            inv = [clause.get_tensor(attributers) for clause in self.clauses]
            self._T = K.any(K.concatenate(inv, axis=1), axis=1)

        return self._T

    def get_executable(self, attributers=None):
        if self._exe is None:
            if self._T is None:
                self.get_tensor(attributers=attributers)
            exe = K.function([self.model.input], [self._T])
            # exe = K.function([self.attributers[0].model.input], [self._T])
            self._exe = lambda x: exe([x])[0]

        return self._exe

    def eval(self, x, attributers=None):
        if self._exe is None:
            self.get_executable(attributers=attributers)
        if np.ndim(x) == K.ndim(self.model.input) - 1:
            x = np.expand_dims(x, 0)
        return self._exe(x)

    def __str__(self):
        s = "\nor\n".join(["(" + str(clause) + ")" for clause in self.clauses])
        s += "\n\t--> Q = {}".format(self.Q)
        s += "\nsupport={:.3}, precision={:.3}".format(self.support, self.precision)
        return s