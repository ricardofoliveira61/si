import numpy as np

from si.data.dataset import Dataset

def train_test_split(dataset:Dataset, test_size:float, random_state:float =123) -> tuple[Dataset,Dataset]:
    """
    Splits a Dataset into training and testing sets.
    
    Parameters
    ----------
    dataset : Dataset
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(dataset.X.shape[0]))
    train_size = int(len(dataset.X.shape[0]) * (1 - test_size))
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    train_dataset = dataset.X[train_indices]
    test_dataset = dataset.X[test_indices]

    return Dataset(train_dataset,y=dataset.y[train_indices],features=dataset.features,label=dataset.label), Dataset(test_dataset,y=dataset.y[test_indices],features=dataset.features,label=dataset.label)
