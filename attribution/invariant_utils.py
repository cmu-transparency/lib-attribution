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
        prec = sum([inv.precision*(inv.support/supp_tot) for inv in Is])
        Ir.append(Invariant(clauses, model, Q=q, support=supp_tot, precision=prec))

    return Ir

def inv_precision(inv, model, x, batch_size=None):
    cl = inv.Q
    inv_evs = inv.eval(x, batch_size=batch_size)
    inv_inds = np.where(inv_evs)[0]
    if len(inv_inds) > 0:
        x_inv = x[inv_inds]
        cl_inds = np.where(model.predict(x_inv, batch_size=batch_size).argmax(axis=1) == cl)[0]
        return float(len(cl_inds))/float(len(x_inv))
    else:
        return 1.

def inv_support(inv, model, x, batch_size=None):
    cl = inv.Q
    x_cl = x[np.where(model.predict(x, batch_size=batch_size).argmax(axis=1) == cl)[0]]
    if len(x_cl) == 0:
        return 1.
    inv_evs = inv.eval(x_cl, batch_size=batch_size)
    inv_inds = np.where(inv_evs)[0]
    return float(len(inv_inds))/float(len(x_cl))

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