import numpy as np

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:#+
    """
    Calculates the Euclidean distance between two sets of points.

    Parameters:
    ...............
    x (np.ndarray):
        A 2D numpy array representing the first set of points. Each row represents a point.#+
    y (np.ndarray):
        A 2D numpy array representing the second set of points. Each row represents a point.#+
    Returns:
    ----------
    np.ndarray:
        A 1D numpy array containing the Euclidean distances between corresponding points in x and y.#+
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1))#+
