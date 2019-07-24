import numpy as np

import keras
import keras.backend as K

from .invariant import Literal, Clause, Invariant


def merge_by_Q(I, model=None):

    if model is None:
        model = I[0].model

    Qs = sorted(list(set([inv.Q for inv in I])))

    Ir = []
    for q in Qs:
        Is = list(filter(lambda inv: inv.Q == q, I))
        clauses = [clause for inv in Is for clause in inv.clauses]
        supp = [inv.support for inv in Is]
        supp_tot = sum(supp)
        prec = sum([inv.precision * (inv.support / supp_tot) for inv in Is])
        Ir.append(Invariant(clauses, model, Q=q,
                            support=supp_tot, precision=prec))

    return Ir


def inv_precision(inv, model, x, batch_size=None):
    cl = inv.Q
    inv_evs = inv.eval(x, batch_size=batch_size)
    inv_inds = np.where(inv_evs)[0]
    if len(inv_inds) > 0:
        x_inv = x[inv_inds]
        cl_inds = np.where(model.predict(
            x_inv, batch_size=batch_size).argmax(axis=1) == cl)[0]
        return float(len(cl_inds)) / float(len(x_inv))
    else:
        return 1.


def inv_support(inv, model, x, batch_size=None):
    cl = inv.Q
    x_cl = x[np.where(model.predict(
        x, batch_size=batch_size).argmax(axis=1) == cl)[0]]
    if len(x_cl) == 0:
        return 1.
    inv_evs = inv.eval(x_cl, batch_size=batch_size)
    inv_inds = np.where(inv_evs)[0]
    return float(len(inv_inds)) / float(len(x_cl))


def tally_total_stats(invs, model, x, batch_size=None):

    classes = list(range(model.output.shape[1]))
    n_per_class = {cls: 0 for cls in classes}
    support = {cls: 0 for cls in classes}
    precision = {cls: 0 for cls in classes}

    for inv in invs:
        n_per_class[inv.Q] += 1

    invs = merge_by_Q(invs)

    for inv in invs:
        support[inv.Q] += inv_support(inv, model, x, batch_size=batch_size)
        precision[inv.Q] += inv_precision(inv, model, x, batch_size=batch_size)

    return n_per_class, support, precision


def get_clause_units(c):
    return set([l.unit for l in c.literals])


def get_invariant_units(inv):
    units = []
    for c in inv.clauses:
        if isinstance(c, Invariant):
            units.extend(list(get_invariant_units(c)))
        elif isinstance(c, Clause):
            units.extend(list(get_clause_units(c)))

    return set(units)


def smooth_logit_tensor_from_literal(l):
    if l.op == K.less_equal:
        return K.expand_dims(
            -(l.attribution_unit - (K.zeros_like(l.attribution_unit, dtype=K.floatx()) + (l.value))), 1)
    elif l.op == K.greater:
        return K.expand_dims(
            l.attribution_unit - (K.zeros_like(l.attribution_unit, dtype=K.floatx()) + float(l.value)), 1)
    else:
        raise ValueError('invalid operator:', l.op)


def smooth_logit_tensor_from_clause(c):
    lit_tensors = [smooth_logit_tensor_from_literal(l) for l in c.literals]
    return sum(lit_tensors) / float(len(lit_tensors))


def smooth_logit_tensor_from_invariant(inv):
    clause_tensors = [smooth_logit_tensor_from_clause(c) for c in inv.clauses]
    return sum(clause_tensors)


def smooth_logit_tensor_from_invariants(invs):
    invs_by_q = merge_by_Q(invs)
    logits = []
    for inv in invs_by_q:
        logits.append(smooth_logit_tensor_from_invariant(inv))
    return K.concatenate(logits, axis=1)


def smooth_probit_tensor_from_invariants(invs, t=10):
    logits = smooth_logit_tensor_from_invariants(invs)
    return K.softmax(logits * t, axis=1)


def smooth_logits_from_invariants(invs):
    t = smooth_logit_tensor_from_invariants(invs)
    logit_fn = K.function([invs[0].model.input], [t])

    def eval_logits(x):
        res = []
        for xi in x:
            res.append(logit_fn([xi[np.newaxis]])[0])
        return np.concatenate(res, axis=0)

    return eval_logits


def smooth_probits_from_invariants(invs, t=10):
    t = smooth_probit_tensor_from_invariants(invs, t=t)
    prob_fn = K.function([invs[0].model.input], [t])

    def eval_probits(x):
        res = []
        for xi in x:
            res.append(prob_fn([xi[np.newaxis]])[0])
        return np.concatenate(res, axis=0)

    return eval_probits


def logit_tensor_from_invariants(invs):
    invs_by_q = merge_by_Q(invs)
    tensors_by_q = [K.expand_dims(
        K.cast(inv.get_tensor(), 'float32'), 1) for inv in invs_by_q]
    logits = K.concatenate(tensors_by_q, axis=1)
    return logits


def probits_from_invariants(invs, t=10):
    logits = logit_tensor_from_invariants(invs)
    probits = K.softmax(logits * t, axis=1)
    probit_fn = K.function([invs[0].model.input], [probits])

    def eval_probits(x):
        res = []
        for xi in x:
            res.append(probit_fn([xi[np.newaxis]])[0])
        return np.concatenate(res, axis=0)

    return eval_probits


def probits_from_multi_invariants(invs_set, t=10):
    logits_set = []
    for invs in invs_set:
        logits_set.append(logit_tensor_from_invariants(invs))
    logits = sum(logits_set)
    probits = K.softmax(logits * t, axis=1)
    input_ph = [invs[0].model.input for invs in invs_set]
    probit_fn = K.function(input_ph, [probits])

    def eval_probits(x):
        res = []
        for xi in x:
            inpt = [xi[np.newaxis] for i in invs_set]
            res.append(probit_fn(inpt)[0])
        return np.concatenate(res, axis=0)

    return eval_probits
