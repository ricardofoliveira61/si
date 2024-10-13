from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH
import os
from si.decomposition.pca import PCA
from si.io.csv_file import read_csv

class TestPCA(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
    
    def test_pca_fit(self):
        self.assertEqual

    def test_pca_transform(self):
        x_reduced = PCA(n_components=2).fit_transform(self.dataset)
        self.assertEqual(x_reduced.shape[1], 2)
        self.assertEqual(x_reduced.shape[0], self.dataset.shape[0])