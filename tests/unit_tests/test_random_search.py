from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.data_file import read_data_file
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression
import numpy as np


class TestRandomGridSearchCV(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_random_grid_search_k_fold_cross_validation(self):

        model = LogisticRegression()

        # setting parameter grid
        # l2_penalty from 1 to 10 with space of 1
        # alpha from 0.001 to 0.0001 with space of -9e-6 
        # max_iter from 1000 to 2000 with space of 5
        parameter_grid= {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200)
        }

        # cross validate the model with 3-fold cross-validation
        # n_iter = 10, which means 10 different combinations of hyperparameters will be tried
        results_ = randomized_search_cv(model= model,
                                        dataset= self.dataset,
                                        hyperparameter_grid= parameter_grid,
                                        cv=3,
                                        n_iter=10)

        # print the results
        self.assertEqual(len(results_["scores"]), 10)

        # get the best hyperparameters
        best_hyperparameters = results_['best_hyperparameters']
        self.assertEqual(len(best_hyperparameters), 3)

        # get the best score
        best_score = results_['best_score']
        self.assertEqual(np.round(best_score, 2), 0.97)
        