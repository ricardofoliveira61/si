from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import stratified_train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy


class TestStackingClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = stratified_train_test_split(self.dataset,test_size=0.3)

    def test_fit(self):

        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()
        
        stacking_classifier = StackingClassifier(models = [knn,logistic_regression,decision_tree], final_model= knn_final)
        stacking_classifier.fit(self.train_dataset)

        self.assertEqual(stacking_classifier.new_dataset.shape()[0], self.train_dataset.shape()[0])
        self.assertEqual(len(stacking_classifier.models), stacking_classifier.new_dataset.shape()[1])



    def test_predict(self):
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()
        
        stacking_classifier = StackingClassifier(models = [knn,logistic_regression,decision_tree], final_model= knn_final)
        stacking_classifier.fit(self.train_dataset)

        predictions = stacking_classifier.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    
    def test_score(self):
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()
        
        stacking_classifier = StackingClassifier(models = [knn,logistic_regression,decision_tree], final_model= knn_final)
        stacking_classifier.fit(self.train_dataset)
        accuracy_ = stacking_classifier.score(self.test_dataset)
        excepted_accuracy = accuracy(self.test_dataset.y,stacking_classifier.predict(self.test_dataset))

        print(round(accuracy_,2))
        self.assertEqual(round(accuracy_, 2), round(excepted_accuracy,2))
