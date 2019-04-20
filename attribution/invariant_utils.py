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