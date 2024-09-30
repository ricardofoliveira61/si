import scipy.stats
from si.data.dataset import Dataset
import scipy

def f_classification (dataset:Dataset) -> tuple:

    """
    

    Parameters
    -----------
    dataset: Dataset
        The Dataset object

    Returns
    -----------
    tuple[float,float]:
        Tuple with F values and tuple with p-values
    
    """

    classes = dataset.get_classes()

    groups = []
    for class_ in classes:
        mask = dataset.y == class_
        group = dataset.X[mask,:]
        groups.append(group)

    return scipy.stats.f_oneway(*groups)
