import mlflow
import logging
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import time
import unittest

try:
    mlflow.autolog()
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading data...")
    data = pd.read_csv("data/Iris.csv")
    data = shuffle(data, random_state=42)
    train_dataset = data.head(100)
    train_dataset.to_csv("data/train_dataset.csv", index=False)
    inference_dataset = data.tail(len(data) - 100)
    inference_dataset.to_csv("data/inference_dataset.csv", index=False)
except FileNotFoundError:
    raise Exception("Data file not found.")

try:
    logging.info("Running training...")
    X = train_dataset.drop(["Id", "Species"], axis=1)
    y = train_dataset["Species"]

    logging.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    start_time = time.time()
    svm = SVC(kernel="linear", probability=True, random_state=0)
    logging.info("Training the model...")
    svm.fit(X_train, y_train)
    end_time = time.time()
    logging.info(f"Train completed in {end_time - start_time} seconds")

    training_accuracy = svm.score(X_train, y_train) * 100
    testing_accuracy = svm.score(X_test, y_test) * 100
    logging.info(f"The accuracy on training data is {training_accuracy}%")
    logging.info(f"The accuracy on test data is {testing_accuracy}%")
except Exception as e:
    raise Exception("An error occurred during training: " + str(e))

try:
    logging.info("Saving the model...")
    filename = "model/model.pkl"
    pickle.dump(svm, open(filename, "wb"))
except FileNotFoundError:
    raise Exception("Model is not saved.")


class Testing(unittest.TestCase):
    def test_training(self):
        self.train_accuracy = training_accuracy
        self.assertGreater(self.train_accuracy, 90, 'Training accuracy is above 90%')

    def test_testing(self):
        self.test_accuracy = testing_accuracy
        self.assertGreater(self.test_accuracy, 90, 'Testing accuracy is above 90%')