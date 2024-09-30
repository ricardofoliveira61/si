import numpy as np

from si.base.transformer import Transformer
from si.statistics.f_classification import f_classification
from si.data.dataset import DataSet

class SelectKBest(Transformer):
    """
    Selects the k best features based on a specified scoring function.
    """

    def __init__(self,k:int, score_func =f_classification, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self,dataset:DataSet)-> "SelectKBest":
        self.F,self.p = self.score_func(dataset)
        return self
    
    def _transform(self, dataset: DataSet) -> DataSet:
        idx = np.argsort(self.F)
        mask = idx[-self.k:]
        new_X = dataset.X[:,mask]
        new_features = dataset.features[mask]

        return DataSet(new_X, dataset.y, features=new_features, label=dataset.label)        
