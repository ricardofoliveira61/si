import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename:str, sep:str = ",", features: bool = False, label: bool = False)-> Dataset :
    """
    Reads a CSV file and returns a Dataset object
    
    Parameters
    ------------
    
    - filename: str
        Name/path of the CSV file to read

    - sep: str
        Delimiter used in the CSV file
    
    - features: bool
        Indicates if the CSV file has features names

    - label: bool
        Indicates if the CSV file has labels. If True, assumes that the label is the last column

    Returns
    ----------
    - Dataset object


    Raises
    -----------
    - ValueError:
        If the seperator given is not a valid separator
    
    - TypeError:
        When the parameters features and label are not boolean
    """

    # if sep not in [",","."]:
    #     raise ValueError("Invalid delimiter. Use either ',' or '.'")
    
    # if type(features) != bool:
    #     raise TypeError("features must be a boolean value")
    
    # if type(label) != bool:
    #     raise TypeError("label must be a boolean value")


    csv =  pd.read_csv(filename, sep=sep)

    if features and label:
        features_names = csv.columns[:-1]
        label = csv.columns[-1]
        x = csv.iloc[:,:-1].to_numpy()
        y = csv.iloc[:,-1].to_numpy()

    elif features and not label:
        features_names = csv.columns
        label = None
        x = csv.to_numpy()
        y = None

    elif label and not features:
        features_names = None
        label = csv.columns[-1]
        x = csv.iloc[:,:-1]
        y = csv.iloc[:,-1]

    else:
        features_names = None
        label = None
        x = csv.to_numpy()
        y = None


    return Dataset(X= x, y= y, features= features_names, label= label)


def write_csv(filename:str, dataset: Dataset, sep:str = ",", features: bool = False, label:bool = False) -> None:
    """
    Writes a CSV file witht the provided data

    Parameters
    -----------

    - filename:str
        Path to save the CSV file
    
    - dataset: Dataset
        Dataset object
    
    - sep:str
        Separator used in the file. By default uses ","
    
    - features: bool
        Whether the file has a header. False by default
    
    - label: bool
        Whether the file has a label. False by default
    """

    data = pd.DataFrame(dataset.X)

    if features:
        data.columns = dataset.features
    
    if label:
        # creation of the column label
        data[dataset.label] = dataset.y
    
    data.to_csv(filename, sep=sep, index= False)
    