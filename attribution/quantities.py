'''
'''


class Qoi(object):
    '''
    Interface for a quantity of interest.
    '''

    def __call__(self, top_of_model):
        '''
        '''
        raise NotImplementedError('This is an abstract method.')

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

    def __call__(self, model):
        return model.output[:, self.c]


class ComparativeQoi(Qoi):
    '''
    '''
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def __call__(self, model):
        output_c1 = (
            model.output[:,c1].mean(axis=1) if isinstance(self.c1, list) else
            model.output[:,c1])
        output_c2 = (
            model.output[:,c2].mean(axis=1) if isinstance(self.c2, list) else
            model.output[:,c2])

        return output_c1 - output_c2     
