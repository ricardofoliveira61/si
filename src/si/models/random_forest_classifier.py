import numpy as np
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.base.model import Model


class RandomForestClassifier(Model):
    """
    Random Forest Classifier is an ensesmble machine learning technique that combines
    multiple decision trees to improve the prediction accuracy and robustness of the model
    and reduce overfitting.
    """

    def __init__(
        self, n_estimators:int = 100, max_features:int = None,
        min_sample_split:int = 5,max_depth:int = 10, mode:str="gini",
        seed: int =123, **kwargs
        ):
        
        """
        Parameters:
        -----------
        n_estimators (int):
            - The number of decision trees in the forest.
        max_features (int): 
            - The maximum number of features to consider when splitting a node.
        min_sample_split (int):
            - The minimum number of samples allowed in a split.
        max_depth (int):
            -The maximum depth of the decision trees in the forest.
        mode (str):
            -The mode to use for calculating the information gain.
        seed (int):
            - The random seed for reproducibility.
        """
        
        # parameters
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        if isinstance(mode,str) and mode.lower() in {"gini","entropy"}:
            self.mode = mode
        else:
            raise ValueError(f'Invalid mode: {mode}. Valid modes are: "gini", "entropy"')
        self.seed = seed

        # attributes
        self.trees = []
        
    
    def _fit(self,dataset:Dataset) ->"RandomForestClassifier":

        """
        Train the decision trees of the random forest

        Parameters:
        -----------
        dataset (Dataset):
            - The dataset to train the random forest on.

        Returns:
        -----------
        self: RandomForestClassifier
          - The trained random forest model.
        """

        # setting the random seed
        np.random.seed(self.seed)

        # defining the max_features if None
        if self.max_features == None:
            self.max_features = int(np.sqrt(dataset.shape()[1]))

        for _ in range(self.n_estimators):
            # creates the bootstrap dataset
            samples = np.random.choice(np.arange(dataset.shape()[0]),size= dataset.shape()[0],replace= True)
            features_idx = np.random.choice(np.arange(dataset.shape()[1]),size= self.max_features, replace= False)
            features = [dataset.features[idx] for idx in features_idx]
            bootstrap_dataset = Dataset(X=dataset.X[samples][:, features_idx],y=dataset.y[samples],features=features,label=dataset.label)

            # fits the bootstrap dataset
            tree = DecisionTreeClassifier(min_sample_split= self.min_sample_split, max_depth=self.max_depth, mode=self.mode)
            tree.fit(bootstrap_dataset)

            # adding the tree to the forest list  (features, tree)
            self.trees.append((tree.dataset.features,tree))


        return self

    def _predict(self, dataset:Dataset)->np.ndarray:

        """
        Predicts the target values for the given dataset using the trained random forest.

        Parameters:
        -----------
        dataset (Dataset):
            - The dataset to predict the target values for.

        Returns:
        -----------
        predictions (np.ndarray):
            - The predicted target values for the given dataset.
        """
        
        # initialize the prections list
        predictions = []
        for features,tree in self.trees:
            # creates a mask to do a subset with only the features used in the tree.
            mask = [True if feature in features else False for feature in dataset.features]
            # creates the subdatset Dataset object with the relevant features only
            X_subet = Dataset(X=dataset.X[:,mask],y=dataset.y,features=features,label=dataset.label)
            # preditcts and append the predictions
            predictions.append(tree.predict(X_subet))
        
        # creates and 2D array. Rows are the samples and columns are the predictions for each tree in the forest
        predictions = np.array(predictions).T

        # for each row find the unique labels and counts the number of times it appears
        label_and_counts = np.apply_along_axis(lambda x: np.unique(x,return_counts=True),axis=1,arr=predictions)
        
        # for each row find the label that appears most often and append it to the predictions list
        label_prediction = np.array([unique[np.argmax(counts)] for unique, counts in label_and_counts])

        print(label_prediction)
        return label_prediction

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        return accuracy(dataset.y, predictions)
    