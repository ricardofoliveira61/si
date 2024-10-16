import numpy as np


def rmse(y_true:np.ndarray, y_pred: np.ndarray) -> float:

    """
    Calculates the Root Mean Squared Error (RMSE) between the true values and predicted values.
    
    Parameters
    ----------
    y_true: np.ndarray
        - An array containing the true values of the label
    
    y_pred: np.ndarray
        - An array containing the predicted values for the label

    Returns
    -------
    float
        - The RMSE value between the true and predicted values
    """
    
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))
