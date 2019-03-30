from .AttributionMethod import AttributionMethod

class VisualizationMethod:

    def __init__(self, attributer, **kwargs):
        assert isinstance(attributer, AttributionMethod), "Need to pass valid attribution method instance"
        self.attributer = attributer
        self.model = attributer.model
        self.layer = attributer.layer

    def visualize(self, x, **kwargs):
        raise NotImplementedError
    
    def visualize_np(self, x, attribs, **kwargs):
        raise NotImplementedError