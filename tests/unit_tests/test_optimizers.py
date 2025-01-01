from unittest import TestCase
import numpy as np
from si.neural_networks.optimizers import Adam, SGD


class TestOptimizers(TestCase):

    def setUp(self):
        
        # create weights and gradients
        self.w = np.random.rand(10000, 1)
        self.grad_loss_w = np.random.rand(10000, 1)

    def test_sgd(self):
        
        optimizer = SGD(learning_rate=0.01)
        updated_w = optimizer.update(self.w, self.grad_loss_w)

        # test if the new weights are close to the expected values
        self.assertTrue(np.allclose(updated_w, self.w - 0.01 * self.grad_loss_w))
        # test if the new weights have the same shape as the original ones
        self.assertEqual(updated_w.shape, self.w.shape)
        # test if the new weights are different from the original ones
        self.assertFalse(np.array_equal(updated_w, self.w))

    def test_adam(self):
        
        optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
        updated_w = optimizer.update(self.w, self.grad_loss_w)
        

        # test if the new weights have the same shape as the original ones
        self.assertEqual(updated_w.shape, self.w.shape)
        # test if the new weights are different from the original ones
        self.assertFalse(np.array_equal(updated_w, self.w))