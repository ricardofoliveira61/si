from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.activation import TanhActivation,SoftmaxActivation
from scipy.special import softmax


class TestSigmoidLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        sigmoid_layer = SigmoidActivation()
        result = sigmoid_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = SigmoidActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestRELULayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        relu_layer = ReLUActivation()
        result = relu_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = ReLUActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestTanhLayer(TestCase):
    
    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_activation_function(self):
        # create a instance of the activation function
        tanh_layer = TanhActivation()
        # compute the activation function output
        result = tanh_layer.activation_function(self.dataset.X)
        # expected result using numpy tanh method
        expected_result = np.tanh(self.dataset.X)

        # test if the output has the same shape as the input
        self.assertEqual(result.shape, self.dataset.shape())
        # test if the output is within the range [-1, 1], tanh function only outputs values between -1 and 1
        self.assertTrue(np.all(np.abs(result) <= 1))
        # test if the output matches the expected result
        self.assertTrue(np.allclose(result,expected_result))

    def test_derivative(self):
        # create a instance of the activation function
        tanh_layer = TanhActivation()
        # compute the derivative of the activation function output
        derivative = tanh_layer.derivative(self.dataset.X)
        # expected result using numpy tanh derivative method
        expected_derivative = 1 - np.square(np.tanh(self.dataset.X))

        # test if the output has the same shape as the input
        self.assertEqual(derivative.shape, self.dataset.shape())
        # test if the derivate has the same values as the expected result
        self.assertTrue(np.allclose(derivative, expected_derivative))


class TestSoftmaxLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
    
    def test_activation_function(self):
        # create a instance of the activation function
        softmax_layer = SoftmaxActivation()
        # compute the activation function output
        result = softmax_layer.activation_function(self.dataset.X)
        # expected result using numpy softmax method
        expected_result = softmax(self.dataset.X, axis=1)
        
        # test if the output has the same shape as the input
        self.assertEqual(result.shape, self.dataset.shape())
        # test if the output matches the expected result
        self.assertTrue(np.allclose(result, expected_result))
        # test if the sum of the output for each row is equal to 1
        self.assertTrue(np.allclose(np.sum(result,axis=1),np.ones(np.sum(result,axis=1).shape)))

    def test_derivative(self):
        # create a instance of the activation function
        softmax_layer = SoftmaxActivation()
        # compute the derivative of the activation function output
        derivative = softmax_layer.derivative(self.dataset.X)
        
        # test if the output has the same shape as the input
        self.assertEqual(derivative.shape, self.dataset.shape())
