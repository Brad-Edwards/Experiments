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
import matplotlib.pyplot as plt
import numpy as np


class DeepNeuralNetwork:
    """
DeepNeuralNetwork Class

A Deep Neural Network (DNN) is an artificial neural network with multiple layers between the input and output layers. These deep networks are capable of modeling complex non-linear relationships. Each layer in a DNN processes the input it receives, transforms it, and passes the processed information to the next layer. This transformation is done through nodes, or "neurons", using weights, bias, and an activation function.

The `DeepNeuralNetwork` class in this module provides an implementation of a DNN for binary classification tasks. The class has methods for loading and preprocessing data, initializing and updating parameters, forward and backward propagation, as well as for training the model and making predictions.

Class Attributes:
    layer_dims (list): A list of integers where the length of the list represents the number of layers in the network and each integer represents the number of units in that layer.
    X_train, X_test, y_train, y_test (numpy.ndarray): The training and testing data, as well as their corresponding labels.
    parameters (dict): The weights and bias of the network, initialized in the method `initialize_parameters`.
    caches (list): List used to store values for backward propagation computed during forward propagation.
    AL (numpy.ndarray): Activations from the last layer, also known as the predictions.

Methods:
    load_data: Load the training and test data from H5 files.
    preprocess_data: Flatten the input images and normalize the pixel values.
    initialize_parameters: Initialize the weights and bias for each layer in the network.
    linear_forward, sigmoid, relu: Different functions used for forward propagation.
    linear_activation_forward: Apply linear transformation and activation function during forward propagation.
    forward_propagation: Perform the full forward propagation for all layers.
    compute_cost: Compute the cost (or loss) of the current prediction.
    linear_backward, relu_backward, sigmoid_backward: Different functions used for backward propagation.
    linear_activation_backward: Apply backward propagation for the linear transformation and activation function.
    backward_propagation: Perform the full backward propagation for all layers.
    update_parameters: Update the weights and bias based on the gradients computed in backward propagation.
    train: Train the model using gradient descent.
    predict: Use the trained model to make predictions on new data.
"""

    def __init__(self, layer_dims):
        """
        Constructor for DeepNeuralNetwork class

        :param layer_dims: A list of integers where the length of the list represents the number of layers in the
        network and each integer represents the number of units in that layer.
        """
        self.X_train_original = None
        self.X_test_original = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.layer_dims = layer_dims

    def load_data(self, train_filepath, test_filepath):
        """
        Load the training and test data from H5 files.
        :param train_filepath: Filepath to the training data H5 file.
        :param test_filepath: Filepath to the test data H5 file.
        :return: The training and test data, as well as their corresponding labels.
        """
        with h5py.File(train_filepath, "r") as f:
            self.X_train = np.array(f["train_set_x"])
            self.y_train = np.array(f["train_set_y"])

        with h5py.File(test_filepath, "r") as f:
            self.X_test = np.array(f["test_set_x"])
            self.y_test = np.array(f["test_set_y"])

        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess_data(self):
        """
        Flatten the input images and normalize the pixel values. For example, a 64x64x3 image will be flattened to a
        vector of shape (12288, 1), where 12288 = 64 x 64 x 3. Each value in the vector will be between 0 and 1.
        Flattening is required to convert the image data to a format that can be used as input to the neural network.
        Converting the pixel values to a range between 0 and 1 helps the model converge faster, because the input
        values are not too far from the output values (which are probabilities between 0 and 1).
        :return: The flattened and normalized training and test data.
        """
        self.X_train = self.X_train.reshape(self.X_train.shape[0], -1).T / 255
        self.X_test = self.X_test.reshape(self.X_test.shape[0], -1).T / 255
        self.y_test = self.y_test.reshape(1, -1)
        self.y_train = self.y_train.reshape(1, -1)
        self.layer_dims.insert(0, self.X_train.shape[0])
        return self.X_train, self.X_test

    def initialize_parameters(self):
        """
        Initialize the weights and bias for each layer in the network. The weights are initialized to random values
        and the bias is initialized to zero. The weights are scaled by a factor of 0.01 to ensure that they are not
        too large. This helps the model converge faster. This value is called the scaling factor, and is a
        hyperparameter that can be tuned to improve the performance of the model.
        :return: A dictionary containing the weights and bias for each layer in the network.
        """
        np.random.seed(1)
        self.parameters = {}
        L = len(self.layer_dims)
        for l in range(1, L):
            self.parameters["W" + str(l)] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]
            ) * 0.05
            self.parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))
            assert self.parameters["W" + str(l)].shape == (
                self.layer_dims[l],
                self.layer_dims[l - 1],
            )
            assert self.parameters["b" + str(l)].shape == (self.layer_dims[l], 1)
        return self.parameters

    def linear_forward(self, A, W, b):
        """
        Compute the linear transformation for a single layer in the network. The linear transformation is computed as
        Z = W * A + b, where W is the weight matrix, A is the activation from the previous layer, and b is the bias
        vector. The linear transformation is used to compute the activation for the next layer. The activation is
        computed using the linear transformation and an activation function. The activation function is applied to
        introduce non-linearity into the network. The activation function used in this model is the ReLU function for
        all layers except the output layer, where the sigmoid function is used. This is because the sigmoid function
        returns a value between 0 and 1, which is useful for binary classification problems. The ReLU function is used
        for all other layers because it is computationally less expensive than the sigmoid function. The ReLU function
        returns a value between 0 and infinity, which is useful for deep neural networks because it helps to prevent
        the vanishing gradient problem.
        :param A: The activation from the previous layer.
        :param W: The weight matrix for the current layer.
        :param b: The bias vector for the current layer.
        :return: The linear transformation and a cache that contains the activation, weight matrix, and bias vector.
        """
        Z = np.dot(W, A) + b
        assert Z.shape == (W.shape[0], A.shape[1])
        cache = (A, W, b)
        return Z, cache

    def sigmoid(self, Z):
        """
        Compute the sigmoid of Z. The sigmoid function is used to compute the activation for the output layer. The
        sigmoid function returns a value between 0 and 1, which is useful for binary classification problems. The
        sigmoid function is defined as A = 1 / (1 + e^(-Z)).
        :param Z: The linear transformation for the output layer.
        :return: The activation for the output layer and a cache that contains the linear transformation.
        """
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache

    def relu(self, Z):
        """
        Compute the ReLU of Z. The ReLU function is used to compute the activation for all layers except the output
        layer. The ReLU function returns a value between 0 and infinity, which is useful for deep neural networks
        because it helps to prevent the vanishing gradient problem. The vanishing gradient problem occurs when the
        gradient becomes very small as the number of layers in the network increases. This makes it difficult to
        update the weights in the earlier layers in the network, which results in slower training. The ReLU function
        is defined as A = max(0, Z).
        :param Z: The linear transformation for the current layer.
        :return: The activation for the current layer and a cache that contains the linear transformation.
        """
        A = np.maximum(0, Z)
        assert A.shape == Z.shape
        cache = Z
        return A, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Compute the linear transformation and activation for a single layer in the network. The linear transformation
        is computed using the linear_forward function. The activation is computed using the sigmoid or ReLU function
        depending on the activation parameter. The activation function is applied to introduce non-linearity into the
        network. Non-linearity is important because it allows the network to learn complex relationships between the
        input and output values.
        :param A_prev: activation from the previous layer
        :param W: weight matrix for the current layer
        :param b: bias vector for the current layer
        :param activation: activation function to use
        :return: the activation and a cache that contains the linear transformation and activation
        """
        A, linear_cache, activation_cache = None, None, None
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        assert A.shape == (W.shape[0], A_prev.shape[1])
        cache = (linear_cache, activation_cache)
        return A, cache

    def forward_propagation(self, X, parameters):
        """
        Compute the forward propagation for the entire network. The forward propagation is computed by computing the
        linear transformation and activation for each layer in the network. The linear transformation is computed
        using the linear_forward function. The activation is computed using the sigmoid or ReLU function depending on
        the activation parameter.
        :param X: input data
        :param parameters: weight matrices and bias vectors for each layer in the network
        :return: the activation for the output layer and a cache that contains the linear transformation and activation
        """
        self.caches = []
        A = X
        L = len(parameters) // 2
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu"
            )
            self.caches.append(cache)
        self.AL, cache = self.linear_activation_forward(
            A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid"
        )
        self.caches.append(cache)
        assert self.AL.shape == (1, X.shape[1])
        return self.AL, self.caches

    def compute_cost(self, AL, Y):
        """
        Compute the cost for the network. The cost is computed using the cross-entropy loss function. The cross-entropy
        loss function is defined as J = -1/m * sum(Y * log(AL) + (1 - Y) * log(1 - AL)). The cross-entropy loss
        function is used for binary classification problems. The cost is computed by summing the cross-entropy loss
        for each training example.
        :param AL: activation for the output layer
        :param Y: true labels for the training examples
        :return: the cost for the network
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)
        assert cost.shape == ()
        return cost

    def linear_backward(self, dZ, cache):
        """
        Compute the gradients for the current layer. The gradients are computed using the linear_backward function.
        The gradients are used to update the weights and biases for the current layer. The gradients are also used to
        compute the gradients for the previous layer. The gradients for the previous layer are computed using the
        linear_backward function. Gradients represent the rate of change of the cost with respect to the weights and
        biases and are computed using the chain rule.
        :param dZ: gradient of the cost with respect to the linear transformation for the current layer
        :param cache: cache that contains the linear transformation for the current layer
        :return: the gradients for the current layer
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape
        return dA_prev, dW, db

    def relu_backward(self, dA, cache):
        """
        The gradient of the ReLU function is defined as dZ = 1 if Z > 0 else 0. The derivative of the ReLU function
        represents the rate of change of the ReLU function with respect to the linear transformation. Intuitively,
        this means that the derivative of the ReLU function represents the rate of change of the activation with
        respect to the linear transformation.
        :param dA: gradient of the cost with respect to the activation for the current layer
        :param cache: cache that contains the linear transformation for the current layer
        :return: the gradients for the current layer
        """
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert dZ.shape == Z.shape
        return dZ

    def sigmoid_backward(self, dA, cache):
        """
        Compute the gradients for the current layer. The gradient of the sigmoid function is defined as
        dZ = dA * s * (1 - s) where s = 1 / (1 + np.exp(-Z)). The derivative of the sigmoid function represents the
        rate of change of the sigmoid function with respect to the linear transformation. Intuitively, this means
        that the derivative of the sigmoid function represents the rate of change of the activation with respect to
        the linear transformation.
        :param dA: gradient of the cost with respect to the activation for the current layer
        :param cache: cache that contains the linear transformation for the current layer
        :return: the gradients for the current layer
        """
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert dZ.shape == Z.shape
        return dZ

    def linear_activation_backward(self, dA, cache, activation):
        """
        Computes the activation for the previous layer. The activation for the previous layer is computed using the
        linear_backward function. The activation for the previous layer is computed by multiplying the gradients for
        the current layer with the derivative of the activation function. The derivative of the activation function
        represents the rate of change of the activation with respect to the linear transformation.
        :param dA: gradient of the cost with respect to the activation for the current layer
        :param cache: cache that contains the linear transformation and activation for the current layer
        :param activation: activation function for the current layer
        :return: the activation for the previous layer
        """
        linear_cache, activation_cache = cache
        dA_prev, dW, db = None, None, None
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):
        """
        Compute the gradients for the network, which are used to update the weights and biases for the network.
        :param AL: activation for the output layer
        :param Y: true labels for the training examples
        :param caches: caches that contain the linear transformations and activations for each layer
        :return: the gradients for the network
        """
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(
            dAL, current_cache, "sigmoid"
        )
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, "relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update the weights and biases for the network using gradient descent. Weights and biases are updated using the
        following equations: W = W - learning_rate * dW and b = b - learning_rate * db where W is the weights, b is the
        biases, dW is the gradient of the weights, db is the gradient of the biases, and learning_rate is the learning
        rate for the network. The learning rate is a hyperparameter that controls how much the weights and biases are
        updated during each iteration of gradient descent.
        :param parameters: parameters for the network
        :param grads: gradients for the network
        :param learning_rate: learning rate for the network
        :return: the updated parameters for the network
        """
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads[
                "dW" + str(l + 1)
            ]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads[
                "db" + str(l + 1)
            ]
        return parameters

    def train(self, X, Y, learning_rate=0.015, num_iterations=3000, print_cost=False):
        """
        Train the network using gradient descent. The network is trained by performing forward propagation and backward
        propagation for a specified number of iterations. The network is trained by performing forward propagation and
        backward propagation for a specified number of iterations.
        :param X: training examples
        :param Y: true labels for the training examples
        :param learning_rate: learning rate for the network
        :param num_iterations: number of iterations for training the network
        :param print_cost: whether to print the cost after every 100 iterations
        :return: the parameters for the network
        """
        np.random.seed(7)
        costs = []
        parameters = self.initialize_parameters()
        for i in range(0, num_iterations):
            AL, caches = self.forward_propagation(X, parameters)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(AL, Y, caches)
            parameters = self.update_parameters(parameters, grads, learning_rate)
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                costs.append(cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per tens)")
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        return parameters

    def predict(self, X, y, parameters):
        """
        Predict the labels for the training examples using the trained network. The labels are predicted by performing
        forward propagation for the training examples using the trained network and then rounding the output of the
        output layer to the nearest integer.
        :param X: data to predict labels for
        :param y: true labels for the data
        :param parameters: parameters for the trained network
        :return: the predicted labels for the data
        """
        m = X.shape[1]
        p = np.zeros((1, m))
        probas, caches = self.forward_propagation(X, parameters)
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        print(f'Accuracy: {np.sum((p == y) / m) * 100}%')
        return p


if __name__ == "__main__":
    dnn = DeepNeuralNetwork([20, 7, 1])
    dnn.load_data("./data/train_catvnoncat.h5", "./data/test_catvnoncat.h5")
    dnn.preprocess_data()
    dnn.train(dnn.X_train, dnn.y_train, num_iterations=5000, print_cost=True)
    dnn.predict(dnn.X_test, dnn.y_test, dnn.parameters)



