import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics import f_classification


class SelectPercentile(Transformer):
    
    def __init__(self, percentile:float, score_func:callable = f_classification,**kwargs):
        """
        Selects features from the given percentile of a score function and returns a new Dataset object with the selected features

        Parameters
        ----------
        percentile: float
            Percentile for selecting features
        
        score_func: callable, optional
            Variance analysis function. Use the f_classification by default for
        """
        super().__init__(**kwargs)
        if isinstance(percentile,int):
            self.percentile = percentile
        else:
            raise ValueError("Percentile must be a integer between 0 and 100")
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self,dataset:Dataset) -> 'SelectPercentile':

        """
        Estimate the F and P values for each feature using the scoring function

        Parameters
        ----------
        dataset: Dataset
            - Dataset object where is intended to select features
        
        Returns
        ----------
        self: object
            - Returns self instance with the F and P values for each feature calculated using the scoring function.
        """

        self.F,self.p = self.score_func(dataset)
        
        return self
    
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects features with the highest F value up to the specified percentile

        Parameters
        ----------
        dataset: Dataset
            - Dataset object where is intended to select features
        
        Returns
        ----------
        dataset: Dataset
            - A new Dataset object with the selected features
        
        """

        # n_feat_select = round(self.percentile/100 * len(self.F))
        # if n_feat_select != 0:

        #     idxs = np.argsort(self.F)[-n_feat_select:]
        #     features = np.array(dataset.features)[idxs]
        #     return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)
    
        # else:
        #     raise ValueError("Cannot select features using the specified percentile")

        feat_select= np.percentile(self.F,100-self.percentile)
        features = np.array(dataset.features)[self.F>=feat_select]
        return Dataset(X=dataset.X[:, self.F>=feat_select], y=dataset.y, features=list(features), label=dataset.label)
