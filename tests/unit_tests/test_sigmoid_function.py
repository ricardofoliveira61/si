from unittest import TestCase
import numpy as np
from si.io.csv_file import read_csv

from si.statistics.sigmoid_function import sigmoid_function



class TestSigmoid(TestCase):

    def setUp(self):

        self.dataset = np.array([[0, 1, 2], [3, 4, 5]])
        

    def test_sigmoid(self):

        predictions = sigmoid_function(self.dataset)
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))

