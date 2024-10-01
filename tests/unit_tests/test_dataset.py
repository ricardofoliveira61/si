import os
import unittest

import numpy as np

from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from datasets import DATASETS_PATH


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())
    
    def test_drop_na(self):
        
        data_w_missing = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris_missing_data.csv'),features=True,label=True)
        data_wo_missing = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris_missing_data.csv'),features=True,label=True)
        data_wo_missing = data_wo_missing.dropna()

        # the data set with missing values must have more rows than the data set without missing values
        self.assertGreater(data_w_missing.X.shape[0],data_wo_missing.X.shape[0])
        self.assertGreater(data_w_missing.y.shape[0],data_wo_missing.y.shape[0])

        # checking the dataset without missing values doesn't have missing values
        self.assertTrue(np.all(np.isnan(data_wo_missing.X)==False))
    
    def test_fill_na(self):
        data_w_missing = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris_missing_data.csv'),features=True,label=True)
        data_wo_missing = data_w_missing.fillna(value=0)

        self.assertTrue(np.all(np.isnan(data_w_missing.X)==False))
        self.assertEqual(data_w_missing.X.shape[0],data_wo_missing.X.shape[0])
        self.assertEqual(data_w_missing.y.shape[0],data_wo_missing.y.shape[0])
        
