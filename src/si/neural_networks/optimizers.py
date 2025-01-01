from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    

class Adam(Optimizer):
    """
    The Adam optimizer is an adaptive learning rate optimization algorithm that combines the best aspects
    of two other popular optimizers: SGD and RMSprop.
    Essentially, Adam maintains an exponentially decaying average of past gradients (first moment) and
    past squared gradients (second moment). This allows it to adapt the learning rate for each parameter 
    individually, accelerating training and often leading to improved performance compared to standard gradient descent.
    
    Adam is known for its efficiency and effectiveness across a wide range of deep learning models.
    It's a popular choice due to its ability to handle sparse gradients and noisy problems, making it a robust and versatile
    optimizer for many machine learning tasks.
    """

    def __init__(self,learning_rate:float,beta_1:float=0.9,beta_2:float=0.999,epsilon:float=1e-8):
        """
        Initialize the Adam optimizer class

        Parameters
        ----------
        learning_rate : float
            - The learning rate used for updating the weights
        beta_1 : float
            - The exponencial decay rate for the 1st moment estimates. Default is 0.9
        beta_2 : float
            - The exponencial decay rate for the 2nd moment estimates. Default is 0.999
        epsilon : float
            - A small constant to prevent division by zero in the denominator. Default is 1e-8
        """
        
        # parameters
        self.learning = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        # attributes
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self,w:np.ndarray,grad_loss_w:np.ndarray)->np.ndarray:
        """
        Updates the weights of a layer

        Parameters
        ----------
        w : np.ndarray
            - The current weights
        grad_loss_w : np.ndarray
            - The gradient of the loss function with respect to the weights

        Returns
        -------
        np.ndarray
            - The updated weights
        """
        # checks if m and v are initialized and if not creates matrices of zeros
        if self.m is None:
            self.m = np.zeros(w.shape)

        if self.v is None:
            self.v = np.zeros(w.shape)

        # update t
        self.t+=1
        
        # compute and update m
        self.m = self.beta_1*self.m + (1-self.beta_1)*grad_loss_w

        # compute and update v
        self.v = self.beta_2*self.v + (1-self.beta_2)*(grad_loss_w**2)

        # compute and update m_hat
        m_hat = self.m / (1 - self.beta_1)

        # compute and update v_hat
        v_hat = self.v / (1 - self.beta_2)

        # compute the moving averages
        updated_weights = w - self.learning * (m_hat / (np.sqrt(v_hat) + self.epsilon))

        return updated_weights
        