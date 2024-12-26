from unittest import TestCase
import numpy as np
from si.data.dataset import Dataset
from si.neural_networks.layers import Dropout


class TestDropOut(TestCase):

    def setUp(self):

        self.data = Dataset.from_random(n_samples= 1000, n_features= 200)

    def test_forward(self):
        # creates a dropout instance with the probability of dropping a neuron equal to 0.5
        dropout = Dropout(probability= 0.5)

        # forward propragation on the training mode
        training = dropout.forward_propagation(self.data.X, training= True)
        # test the shape of the dropout output and the input data
        self.assertEqual(training.shape, self.data.X.shape)
        # test if the mean of the mask is equal to the probability of keeping a neuron
        self.assertAlmostEqual(np.mean(dropout.mask), 0.5,places=1)
        # test if the output of the dropout layer has values equal to zero
        self.assertTrue(np.any(training == 0))
        # test if the output of the dropout layer is different from the original input
        self.assertFalse(np.array_equal(training, self.data.X))

        # forward propragation on the inferance mode
        inference = dropout.forward_propagation(self.data.X)
        # test if the output of the dropout layer is equal to the original input data in inferance mode
        self.assertTrue(np.array_equal(inference,self.data.X))

    def test_backward(self):
        # creates a dropout instance with the probability of dropping a neuron equal to 0.5
        dropout= Dropout(probability= 0.5)
        # performs forward propagation
        dropout.forward_propagation(self.data.X, training=True)
        # creates a random output error to represent the error signal backpropagated from the subsequent layer
        output_error = np.random.random(self.data.shape())
        # backward propragation
        input_error = dropout.backward_propagation(output_error= output_error)

        # test if the shape of input and output errors match
        self.assertTrue(output_error.shape,input_error.shape)

        # test if the input error computed matches the expected error
        input_error_expected = output_error * dropout.mask
        self.assertTrue(np.allclose(input_error, input_error_expected))