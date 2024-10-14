import numpy as np


def mse(y_true: np.ndarray, y_pred:np.ndarray) -> float:
    """
    Calculates the mean squared error between true and predicted values of a label.

    Parameters
    ----------
    y_true: np.ndarray
        - True values of the label.
    y_pred: np.ndarray
        - Predicted values of the label.
    
    Returns
    -------
    mse_value: float
        - Mean squared error between y_true and y_pred.
    """

    mse_value = 1/len(y_true) * np.sum((y_true - y_pred)**2)
    return mse_value
