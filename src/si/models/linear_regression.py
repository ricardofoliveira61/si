import numpy as np
from base.model import Model
from data.dataset import Dataset


class RidgeRegression(Model):


    def __init__ (self, l2_penalty:float = 1,alpha: float = 0.001,max_iter:int = 100,patience:int = 5,scale:bool = True, **kwargs):

        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = 0
        self.mean = None
        self.std = None
        self.cost_history = {}
    

    def _fit(self, dataset:Dataset) -> 'RidgeRegression':
        if self.scale:
            self.mean = dataset.get_mean()
            self.std = np.std(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std

        else:
            X = dataset.X

        self.theta = np.zeros(X.shape[1])

        i = 0
        early_stopping = 0

        while i< self.max_iter and early_stopping < self.patience:

            y_pred = np.dot(self.theta, X) + self.theta_zero

            gradient = (self.alpha /)
        
        
        

    def _predict(self):
        pass

    def _score(self, dataset: Dataset, predictions: Dataset) -> float:
        pass

    def cost(self):
        pass