'''
'''

class Qoi(object):
    '''
    Interface for a quantity of interest.
    '''

    def __init__(self, qoi_fn):
        self.qoi_fn = qoi_fn

    def __call__(self, g):
        '''
        '''
        return self.qoi_fn(g)

    @staticmethod
    def for_class(c):
        '''
        '''
        return ClassQoi(c)

    @staticmethod
    def comparative(positive_class, negative_class):
        '''
        '''
        return ComparativeQoi(positive_class, negative_class)


class ClassQoi(Qoi):
    '''
    '''
    def __init__(self, c, relative=False):
        self.c = c

    def __call__(self, g):
        return g.output[:, self.c]


class ComparativeQoi(Qoi):
    '''
    '''
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def __call__(self, g):
        output_c1 = (
            g.output[:,self.c1].mean(axis=1) if isinstance(self.c1, list) else
            g.output[:,self.c1])
        output_c2 = (
            g.output[:,self.c2].mean(axis=1) if isinstance(self.c2, list) else
            g.output[:,self.c2])

        return output_c1 - output_c2     
