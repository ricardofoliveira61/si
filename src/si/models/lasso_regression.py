import numpy as np
from base.model import Model
from data.dataset import Dataset
from si.metrics.mse import mse


class LassoRegression(Model):
    """
    Lasso Regression is a type of linear regression analysis that uses L1 regularization. The L1 regularization is a penalty that is add to the 
    loss function that is proportional to the absolute value of the coefficients. This penalty term encourages the model to shrink or even
    eliminate coefficients of features that are not important for predicting the target variable.
    """

    def __init__(self, l1_penalty:float, max_iter: int = 1000, patience: int = 5, scale:bool = True, **kwargs):

        """
        Initialize the Lasso Regression model

        Parameters
        ----------
        l1_penalty : float
            - L1 regularization parameter

        alpha : float
            - learning rate

        max_iter : int
        """
        
        # parameters
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}


    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        
        if self.scale:
            # compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        i = 0
        early_stopping = 0

        # gradient descent
        while i < self.max_iter and early_stopping < self.patience:
            # predicted y
            y_pred = np.dot(X, self.theta) + self.theta_zero

            for feature in range(n):

                # compute the residuals
                residuals = y_pred - dataset.y[:, feature]

                # compute the penalty
                penalization_term = self.theta[feature] * (1 - self.alpha * (self.l1_penalty / m))

                # updating the model parameters
                self.theta[feature] = penalization_term - self.alpha * np.dot(residuals, X[:, feature]) / m


            # # computing and updating the gradient with the learning rate
            # gradient = (self.alpha / m) * np.dot(y_pred - dataset.y, X)

            # computing the penalty
             # what is this ??????
            # penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))
            # penalty_term = np.sign(self.theta) * (np.abs(self.theta) - self.l1_penalty * self.alpha / m)

            # updating the model parameters
            # features thetas
            self.theta = self.soft_threshold(residuals,self.l1_penalty)/
            # self.theta = penalization_term - gradient

            # theta zero
            self.theta_zero = (1/n)*np.sum(dataset.y) - np.dot(self.theta, np.sum(X,axis=0)/n)
            # self.theta_zero = (1/n)*np.sum(dataset.y) - np.dot(np.sum(self.theta), np.sum(X,axis=0)/n)
            # is this formula right?

            # compute the cost
            self.cost_history[i] = self.cost(dataset)
            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0
            i += 1

        return self

    
    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) for the lasso regression  model on the dataset using L1 regularization

        Parameters
        ----------
        dataset : Dataset
            - the dataset to compute the cost function

        Returns
        -------
        cost : float
            - the cost function value
        """

        y_pred = self.predict(dataset)

        return 1/(2*len(dataset.y)) * np.sum((dataset.y - y_pred) ** 2) + self.l1_penalty * np.sum(np.abs(self.theta))


    def _predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        predictions: np.ndarray
            Predictions

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        return mse(dataset.y, predictions)

    
    def soft_threshold(self, residual, l1_penalty: float) -> np.ndarray:
        pass
   

