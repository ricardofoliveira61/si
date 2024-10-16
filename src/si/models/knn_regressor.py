from typing import Callable, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    KNN regression is a non-parametric machine learning method suitable for regression problems.
    This method classifies new sample based on a similarity measure, predicting the class of
    the new sample by looking at the values of the k-nearest samples in the training data.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN classifier

        Parameters
        ----------
        k: int
            The number of k nearest example to consider
        distance: Callable
            Function that calculates the distance between a sample and the samples
            in the training dataset
        """

        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to (training dataset)

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _get_closest_value(self, sample: np.ndarray) -> Union[int, float]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest value of

        Returns
        -------
        value: int or float
            The closest value
        """

        # compute the distance between the sample and the training dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the values of the k nearest neighbors
        k_nearest_neighbors_label_values = self.dataset.y[k_nearest_neighbors]

        # get the average value of the k nearest neighbors
        value = np.sum(k_nearest_neighbors_label_values) / self.k

        return value

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the label values of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the values of (testing dataset)

        Returns
        -------
        predictions: np.ndarray
            An array of predicted values for the testing dataset
        """

        # compute the predictions for each row(sample) of the testing dataset
        predictions = np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset, predictions:np.ndarray) -> float:

        """
        Computes the root mean squared error between the estimated values and the true values of a given dataset

        Parameters
        ----------
        dataset: Dataset
            - The dataset to evaluate the model on
        
            
        Returns
        -------
        float
            - Correspondes to the root mean squared error of the model for the given dataset
        """

        return rmse(dataset.y,predictions)
    


