from si.data.dataset import Dataset
from si.base.transformer import Transformer

class VarianceThreshold(Transformer):
    
    def __init__(self, treshold, **kwargs):
        super().__init__(**kwargs)
        self.treshold = treshold
        self.variance = None

    def _fit(self, dataset: Dataset):

        self.variance = dataset.get_variance()

    def _transform(self, dataset: Dataset) ->Dataset:

        mask = self.variance > self.treshold

        new_X = dataset.X[:,mask]
        new_features = dataset.features[mask]

        return Dataset(new_X, dataset.y, features= new_features, label=dataset.label)