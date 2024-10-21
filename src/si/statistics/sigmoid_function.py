
import numpy as np


def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    Calculates the sigmoid function of X

    Parameters
    ----------
    X : np.ndarray
        Input array

    Returns
    -------
    sigmoid_values : np.ndarray
        Array with sigmoid values
    """
    
    return 1 / (1 + np.exp(-X))