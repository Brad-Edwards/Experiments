# This file contains a simple implementation of Logistic Regression using Gradient Descent.
#
# Copyright (c) 2023 Brad Edwards
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class LogisticRegressionGD:
    """
    A simple implementation of Logistic Regression using Gradient Descent.
    """

    def __init__(
        self, learning_rate=0.01, num_iterations=1000, scale_range=(0, 1), batch_size=1
    ):
        """
        Initializes a Logistic Regression model using Gradient Descent for optimization.

        Args:
        learning_rate (float): The rate at which the model learns and updates its weights during training.
        num_iterations (int): The number of iterations the model will go through during training.
        scale_range (tuple): The range to which the features will be normalized.
        batch_size (int): The number of samples to be passed through the model at once during training.
        This can be adjusted depending on the size of the dataset and memory limitations.

        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.scale_range = scale_range
        self.batch_size = batch_size

    def _sigmoid(self, x):
        """
        Computes the sigmoid function, which is used as the activation function in this Logistic Regression model. The
        sigmoid function squashes input values to a range between 0 and 1, which can be interpreted as the probability
        of the positive class in a binary classification task.

        Args:
        x (float): The input to the sigmoid function.

        Returns:
        float: The output of the sigmoid function, a value between 0 and 1.
        """
        return 1 / (1 + np.exp(-x))

    def flatten(self, X):
        """
        Flattens the input data, which is a 3D array of shape (num_samples, height, width), into a 2D array of shape
        (num_samples, height * width). This is necessary because the input data is passed through a fully connected
        layer, which requires a 2D input.

        Args:
        X (numpy array): The input data to be flattened.

        Returns:
        numpy array: The flattened input data.
        """

        return X.reshape(X.shape[0], -1)

    def normalize(self, X):
        """
        Normalizes the input data using the MinMaxScaler, which transforms features by scaling each feature
        to a given range. This is an important preprocessing step in deep learning, as it ensures that all input
        features are on a similar scale, which can help the model learn more effectively.

        Args:
        X (numpy array): The input data to be scaled.

        Returns:
        numpy array: The scaled input data.
        """
        return MinMaxScaler(feature_range=self.scale_range).fit_transform(X)

    def _generate_batches(self, X, y):
        """
        Generates batches from the input data. These batches are used during training, allowing the model
        to update its weights using a subset of the data at each step, which can be more computationally
        efficient and can also help prevent over-fitting.

        Args:
        X (numpy array): The input data.
        y (numpy array): The corresponding labels.

        Yields:
        tuple: A tuple containing a batch of input data and corresponding labels.
        """
        assert X.shape[0] == y.shape[0]
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        for i in range(0, X.shape[0], self.batch_size):
            yield X[i : i + self.batch_size], y[i : i + self.batch_size]

    def _cost_function(self, y_true, y_pred):
        """
        Calculates the cost function, a key concept in machine learning and deep learning used for
        evaluating the performance of the model during training. The cost function quantifies the error
        between predicted values and expected values and presents it in the form of a single real number.

        For logistic regression, we use the binary cross-entropy (or log loss) as the cost function.
        The goal of training is to find the model parameters (weights and bias in this case) that
        minimize this cost function.

        Args:
        y_true (numpy array): The actual labels for the data.
        y_pred (numpy array): The labels predicted by the model.

        Returns:
        float: The computed cost, a single real number representing the total error of our model's predictions.
        """
        num_samples = y_true.shape[0]
        return (-1 / num_samples) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    def train(self, X, y):
        """
        Trains the Logistic Regression model using the provided training data. The training process involves several steps:

        - First, the input data is flattened and normalized, ensuring it has the correct shape and scale for the model to process.

        - The model's weights and bias are initialized. In this case, weights are initialized to zero, but other initialization strategies can also be used.

        - The model then enters a loop that runs for a specified number of iterations. Each iteration represents a complete pass (both forward and backward)
        through the entire dataset, which is divided into batches of a certain size.

        - In each iteration, for each batch, the model first calculates its predictions (y_predicted) using the current weights and bias, and the sigmoid function
        (forward propagation).

        - Then it calculates the gradients of the weights (dw) and bias (db) with respect to the cost function (backward propagation). The gradients represent
        the rate of change of the cost with respect to the parameters, and they are used to update the parameters in the direction that decreases the cost.

        - The weights and bias are then updated using the calculated gradients and the learning rate. The learning rate controls how big of a step we take in
        the direction indicated by the gradients.

        - Finally, the cost for the current iteration is calculated and printed every 100 iterations to provide insight into the training process.

        Args:
        X (numpy array): The input data for training the model.
        y (numpy array): The true labels corresponding to the input data.

        Prints:
        str: A string representing the cost at every 100th iteration, providing insight into the model's learning progress.

        """
        X = self.flatten(X)
        X = self.normalize(X)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.num_iterations):
            for X_batch, y_batch in self._generate_batches(X, y):
                num_samples = X_batch.shape[0]
                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_predicted = self._sigmoid(linear_model)

                dw = (1 / num_samples) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1 / num_samples) * np.sum(y_predicted - y_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                cost = self._cost_function(y_batch, y_predicted)

            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X, y_true):
        """
        Predicts class labels for the input data using the trained Logistic Regression model. The model
        uses its learned weights and bias to make predictions. The output of the model is passed through
        the sigmoid function to squash the output to a range.
        """
        X = self.flatten(X)
        X = self.normalize(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_classes = (y_predicted > 0.5).astype(int)

        print(
            f"\nAccuracy score: {accuracy_score(y_true, y_predicted_classes) * 100}%\n"
        )

        print("Classification Report:")
        print(
            classification_report(
                y_true, y_predicted_classes, target_names=["Non-Cat", "Cat"]
            )
        )

    def load_csv(self, filepath, target_column, test_size=0.2, random_state=42):
        """
        Load data from a CSV file and split it into training and test sets.
        Note: This method assumes that the input data is in CSV format.

        Args:
        filepath (str): The path to the CSV file to load.
        target_column (str): The name of the column in the CSV file that should be used as the target variable.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.

        Returns:
        tuple: A tuple containing four arrays: the training data, the test data, the training labels, and the test labels.
        """
        # Load the data from the CSV file
        data = pd.read_csv(filepath)

        # Separate the target variable from the features
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Return the split data
        return X_train, X_test, y_train, y_test

    def load_h5(self, train_filepath, test_filepath):
        """
        Load training and test data from H5 files.

        Args:
        train_filepath (str): The path to the training H5 file.
        test_filepath (str): The path to the test H5 file.

        Returns:
        tuple: A tuple containing two arrays: the training data and the test data.
        """
        # Load the training data from the H5 file
        with h5py.File(train_filepath, "r") as f:
            X_train = np.array(f["train_set_x"])
            y_train = np.array(f["train_set_y"])

        # Load the test data from the H5 file
        with h5py.File(test_filepath, "r") as f:
            X_test = np.array(f["test_set_x"])
            y_test = np.array(f["test_set_y"])

        # Return the loaded data
        return X_train, X_test, y_train, y_test


def main():
    model = LogisticRegressionGD(learning_rate=0.01, num_iterations=1000, batch_size=32)

    X_train, X_test, y_train, y_test = model.load_h5(
        "./data/train_catvnoncat.h5", "./data/test_catvnoncat.h5"
    )

    model.train(X_train, y_train)

    model.predict(X_test, y_test)


if __name__ == "__main__":
    main()
