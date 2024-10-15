from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.decomposition.pca import PCA
from si.io.csv_file import read_csv
from sklearn.decomposition import PCA as PCA_sk
import pandas as pd


class TestPCA(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.dataset_sk = pd.read_csv(self.csv_file)
    
    def test_pca_fit(self):
        iris_fit = PCA(n_components=2).fit(self.dataset)
        iris_fit_sklearn = PCA_sk(n_components=2).fit(self.dataset_sk.iloc[:, :4])

        self.assertTrue(np.allclose(iris_fit.explained_variance, iris_fit_sklearn.explained_variance_ratio_))
        self.assertEqual(len(iris_fit.explained_variance),2)
        self.assertEqual(len(iris_fit.components),2)
        self.assertTrue(np.allclose(iris_fit.mean, iris_fit_sklearn.mean_))
        
        

    def test_pca_transform(self):
        x_reduced = PCA(n_components=2).fit_transform(self.dataset)
        self.assertEqual(x_reduced.shape()[1], 2)
        self.assertEqual(x_reduced.shape()[0], self.dataset.shape()[0])
        