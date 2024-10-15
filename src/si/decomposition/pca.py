import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class PCA(Transformer):

    def __init__(self, n_components, **kwargs):
        """
        Principal Component Analysis (PCA)

        Parameters
        ----------
        n_components: int
            Number of components.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.is_fitted = False
        self.mean = None
        self.covariance = None
        self.e_values = None
        self.components = None
        self.explained_variance = None
        self.e_vectores = None
    
    def _fit(self, dataset:Dataset) -> "PCA":
        """
        Estimates the mean, principal componentes and the explained variance.

        Parameters
        ----------
        dataset: Dataset
            Dataset object used to estimate the PCA parameters.

        Returns
        -------
        self: PCA

        Raises
        -------
        ValueError;
            If n_componets is 0 or greater than the number of features.
        """

        if self.n_components == 0 or self.n_components> dataset.shape()[1]:
            raise ValueError("n_components must be a positive integer less than or equal to the number of features.")

        # centering the data
        self.mean = dataset.get_mean()
        dataset.X = dataset.X - self.mean


        # computing the covariance matrix of the centered data and eigenvalue decomposition on the covariance matrix
        # rowvar = False ensures that the columns of the dataset are intrepreted as variables
        self.covariance = np.cov(dataset.X, rowvar= False)
        self.e_values, self.e_vectores = np.linalg.eig(self.covariance)
        # garantees real eigenvalues since numerical approximations or rounding errors can lead to complex eigenvalues on a real valued covariance matrix
        self.e_values = np.real(self.e_values)


        # infer the principal components, sorting them by descending order of eigenvalues
        principal_components_idx = np.argsort(self.e_values) [-self.n_components:][::-1]
        

        # Infer the explained variance
        self.explained_variance = self.e_values[principal_components_idx] / np.sum(self.e_values)
        self.components = self.e_vectores[:, principal_components_idx].T

        self.is_fitted = True

        return self
        

    def _transform(self, dataset:Dataset)-> Dataset:
        """
        Tranforms the intended dataset to the principal components.

        Parameters
        ----------
        dataset: Dataset
            Dataset object to be transformed.
        
        Returns
        -------
        Dataset
            Dataset object with the transformed features.        
        """

        # centering the dataset
        X_centered = dataset.X - self.mean

        # reducing the dataset to the principal components
        X_reduced = np.dot(X_centered, self.components.T)

        return Dataset(X_reduced, y= dataset.y, features=[f"PC{i+1}" for i in range(self.n_components)], label= dataset.label)
    
    def get_covariance(self)-> np.ndarray:
        """
        Retunrs the covariance matrix of the centered data.

        Returns
        -------
        np.ndarray
            - Covariance matrix

        Raises
        -------
        ValueError
            - If PCA has not been fitted to your data.
        """

        if not self.is_fitted:
            raise ValueError("PCA has not been fitted to your data.")
        
        else:
            return self.covariance


        
