import numpy as np

import keras
import keras.backend as K
import copy

class Literal(object):

    def __init__(self, layer, unit, value, attribution_unit=None):
        self.layer = layer
        self.unit = unit
        self.value = value
        self.attribution_unit = attribution_unit

    def get_tensor(self, attribution_unit=None):
        u_t = attribution_unit if attribution_unit is not None else self.attribution_unit
        assert u_t is not None, "Need to pass attribution unit tensor either directly or in constructor"

        zeros = K.zeros_like(u_t)
        ones = K.ones_like(u_t)
        eq_fn = K.not_equal if self.value == 1 else K.equal
        u_bt = K.expand_dims(K.switch(eq_fn(u_t, zeros), ones, zeros), 1)

        return u_bt

    def __deepcopy__(self, memo):
        return copy.copy(self)

    def __str__(self):
        return "{}[{}] = {}".format(self.layer.name, self.unit, self.value)

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

    def add_clause(self, clause, copy_self=False):
        t = copy.deepcopy(self) if copy_self else self
        t.clauses.append(clause)
        return t

    def get_tensor(self, attributers=None):
        inv = [clause.get_tensor(attributers) for clause in self.clauses]

        return K.any(K.concatenate(inv, axis=1), axis=1)

    def get_executable(self, attributers=None):
        t = self.get_tensor(attributers=attributers)
        exe = K.function([self.model.input], [t])

        return lambda x: exe([x])[0]

    def __str__(self):
        s = "\nor\n".join(["(" + str(clause) + ")" for clause in self.clauses])
        s += "\n\t--> Q = {}".format(self.Q)
        s += "\nsupport={:.3}, precision={:.3}".format(self.support, self.precision)
        return s