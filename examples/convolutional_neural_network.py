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

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from scipy.signal import convolve2d


class ConvLayer:
    """
    The ConvLayer (Convolutional Layer) class represents a layer in a convolutional neural network (CNN).
    Convolutional layers are fundamental to many types of neural networks that process grid-like topology data,
    such as images.

    In a convolutional layer, filters (also known as kernels) are convolved with the input data to produce a set
    of output values in an operation known as a convolution. This process can identify features or patterns in
    the input data.

    The ConvLayer class includes methods to perform the forward pass (computing the output values) and the
    backward pass (computing the gradients).

    Attributes:
        num_filters (int): The number of filters in the convolutional layer.
        filter_size (int): The size of each filter. Filters are assumed to be square.
        filters (numpy.ndarray): The set of filters to be used in the convolution operation.
        last_input (numpy.ndarray): The last input data processed by the layer. Stored for use in the backward pass.

    Methods:
        forward(input): Processes the input data through the layer, performing the convolution operation,
                        and returns the output.
        backward(grad_output): Uses the stored input and the gradient of the output to compute the gradient of
                               the input and the layer's filters.
    """

    def __init__(self, num_filters, filter_size, num_input_channels):
        """
        Initializes the ConvLayer with a set of random filters.
        Args:
            num_filters (int): The number of filters in the layer.
            filter_size (int): The size of each filter.
            num_input_channels (int): The number of channels in the input data. For example, RGB images have 3 channels.
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(
            num_filters, filter_size, filter_size, num_input_channels
        ) / (filter_size**2 * num_input_channels)
        self.last_input = None

    def forward(self, input):
        """
        Processes the input data through the layer, performing the convolution operation.
        Args:
            input (numpy.ndarray): The input data to the layer.
        Returns:
            numpy.ndarray: The output of the layer after applying the filters to the input.
        """
        self.last_input = input
        h, w, _ = input.shape
        output = np.zeros(
            (h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters)
        )
        for f in range(self.num_filters):
            for c in range(input.shape[-1]):
                output[:, :, f] += convolve2d(
                    input[:, :, c], self.filters[f, :, :, c], mode="valid"
                )
        return output

    def backward(self, grad_output):
        """
        Uses the stored input and the gradient of the output to compute the gradient of the input and the layer's filters.
        Args:
            grad_output (numpy.ndarray): The gradient of the output.
        Returns:
            numpy.ndarray: The gradient of the input.
        """
        input = self.last_input
        grad_filters = np.zeros(self.filters.shape)
        grad_input = np.zeros(input.shape)
        for f in range(self.num_filters):
            for c in range(input.shape[-1]):
                grad_filters[f, :, :, c] = convolve2d(
                    input[:, :, c], grad_output[:, :, f], mode="valid"
                )
                grad_input[:, :, c] += convolve2d(
                    grad_output[:, :, f],
                    np.flip(self.filters[f, :, :, c], axis=(0, 1)),
                    mode="full",
                )
        self.filters -= 0.01 * grad_filters
        return grad_input


class MaxPoolLayer:
    """
    The MaxPoolLayer class represents a max pooling layer in a convolutional neural network (CNN).
    Max pooling is a sample-based discretization process, aimed at reducing the dimensionality of images
    by selecting the maximum value of particular image sections.

    In the context of CNNs, max pooling operation is used to decrease the computational power required to
    process the data through dimensionality reduction. Furthermore, it also helps prevent overfitting by
    providing an abstracted form of the representation.

    The MaxPoolLayer class includes methods to perform the forward pass (computing the output values) and the
    backward pass (computing the gradients).

    Attributes:
        pool_size (int): The size of the max pooling window. The window is assumed to be square.
        last_input (numpy.ndarray): The last input data processed by the layer. Stored for use in the backward pass.

    Methods:
        forward(input): Processes the input data through the layer, performing the max pooling operation,
                        and returns the output.
        backward(grad_output): Uses the stored input and the gradient of the output to compute the gradient of
                               the input.
    """

    def __init__(self, pool_size):
        """
        Initializes the MaxPoolLayer with a specific pool size.
        Args:
            pool_size (int): The size of the max pooling window.
        """
        self.pool_size = pool_size
        self.last_input = None

    def forward(self, input):
        """
        Processes the input data through the layer, performing the max pooling operation.
        Args:
            input (numpy.ndarray): The input data to the layer.
        Returns:
            numpy.ndarray: The output of the layer after performing max pooling operation on the input.
        """
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))
        for i in range(h // self.pool_size):
            for j in range(w // self.pool_size):
                for f in range(num_filters):
                    output[i, j, f] = np.max(
                        input[
                            i * self.pool_size : (i + 1) * self.pool_size,
                            j * self.pool_size : (j + 1) * self.pool_size,
                            f,
                        ]
                    )
        return output

    def backward(self, grad_output):
        """
        Uses the stored input and the gradient of the output to compute the gradient of the input.
        Args:
            grad_output (numpy.ndarray): The gradient of the output.
        Returns:
            numpy.ndarray: The gradient of the input.
        """
        input = self.last_input
        h, w, num_filters = input.shape
        grad_input = np.zeros(input.shape)
        for i in range(h // self.pool_size):
            for j in range(w // self.pool_size):
                for f in range(num_filters):
                    mask = input[
                        i * self.pool_size : (i + 1) * self.pool_size,
                        j * self.pool_size : (j + 1) * self.pool_size,
                        f,
                    ] == np.max(
                        input[
                            i * self.pool_size : (i + 1) * self.pool_size,
                            j * self.pool_size : (j + 1) * self.pool_size,
                            f,
                        ]
                    )
                    grad_input[
                        i * self.pool_size : (i + 1) * self.pool_size,
                        j * self.pool_size : (j + 1) * self.pool_size,
                        f,
                    ] = (
                        np.repeat(
                            np.repeat(grad_output[i, j, f], self.pool_size),
                            self.pool_size,
                        ).reshape(self.pool_size, self.pool_size)
                        * mask
                    )  # repeat the grad_output
        return grad_input


class FullyConnectedLayer:
    """
    The FullyConnectedLayer class is an implementation of a fully connected layer (or a dense layer) in a neural network.

    Fully connected layers are a type of layer in a neural network where each neuron in the layer is connected to every neuron in the previous layer. Each connection has an associated weight which is learned during the training process. The purpose of a fully connected layer is to learn non-linear combinations of the high-level features as represented by the output of the previous layer.

    Attributes:
    - last_input (numpy.ndarray): The last input data that was passed to the forward method. Stored for use in the backward pass.
    - last_input_shape (tuple): The shape of the last input data that was passed to the forward method. Stored for use in the backward pass.
    - weights (numpy.ndarray): The weights associated with the connections in this layer. These are learned during the training process.
    - bias (numpy.ndarray): The bias units associated with the neurons in this layer. These are also learned during the training process.
    - output_dim (int): The dimensionality of the output from this layer.

    Methods:
    - forward(input): This method takes the input data and applies the layer operations to it. The input data is first flattened (if it is not already), then the dot product of the flattened input and the weights is calculated and the bias is added to generate the output values.
    - backward(grad_output): This method performs the backward pass of the backpropagation algorithm. It calculates the gradients of the weights and biases with respect to the loss function, and then updates the weights and biases using a simple gradient descent update rule.
    """

    def __init__(self, output_dim):
        """
        Initializes a new instance of the FullyConnectedLayer class.

        Args:
        - output_dim (int): The number of neurons in this layer, i.e., the dimensionality of the output from this layer.
        """
        self.last_input = None
        self.last_input_shape = None
        self.weights = None
        self.bias = None
        self.output_dim = output_dim

    def forward(self, input):
        """
        Perform the forward pass through the layer.

        Args:
        - input (numpy.ndarray): The input data for this layer. This can be the raw input data for the network or the output from the previous layer.

        Returns:
        - numpy.ndarray: The output from this layer.
        """
        if self.weights is None:
            self.weights = np.random.randn(input.size, self.output_dim) / input.size
            self.bias = np.zeros(self.output_dim)
        self.last_input_shape = input.shape
        input_flattened = input.flatten()
        self.last_input = input_flattened
        output_values = np.dot(input_flattened, self.weights) + self.bias
        return output_values

    def backward(self, grad_output):
        """
        Perform the backward pass through the layer.

        Args:
        - grad_output (numpy.ndarray): The gradient of the loss function with respect to the output of this layer. This is computed in the subsequent layer during the backpropagation algorithm.

        Returns:
        - numpy.ndarray: The gradient of the loss function with respect to the input of this layer. This will be passed to the previous layer during the backpropagation algorithm.
        """
        input = self.last_input
        grad_bias = grad_output.sum(axis=0)
        grad_weights = np.outer(self.last_input, grad_output)
        self.weights -= 0.01 * grad_weights
        self.bias -= 0.01 * grad_bias
        return np.dot(grad_output, self.weights.T).reshape(self.last_input_shape)


class ReLULayer:
    """
    The ReLULayer class is an implementation of the rectified linear unit (ReLU) layer in a neural network.

    The rectified linear unit (ReLU) is a type of activation function that is widely used in neural networks and deep learning models. The function returns 0 if the input is negative, and the input itself if the input is equal to or more than 0.

    The purpose of the ReLU function is to introduce non-linearity into the network, allowing the network to learn complex patterns in the data. It is a computationally efficient function, as it simply thresholds the input at zero.

    Attributes:
    - last_input (numpy.ndarray): The last input data that was passed to the forward method. Stored for use in the backward pass.

    Methods:
    - forward(input): This method takes the input data and applies the ReLU function to it.
    - backward(grad_output): This method performs the backward pass of the backpropagation algorithm. It computes the gradients of the input data with respect to the loss function and multiplies it with the incoming gradients.
    """

    def __init__(self):
        """
        Initializes a new instance of the ReLULayer class.
        """
        self.last_input = None

    def forward(self, input):
        """
        Perform the forward pass through the layer by applying the ReLU function to the input data.

        Args:
        - input (numpy.ndarray): The input data for this layer. This can be the raw input data for the network or the output from the previous layer.

        Returns:
        - numpy.ndarray: The output from the ReLU function applied to the input data.
        """
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        """
        Perform the backward pass through the layer.

        Args:
        - grad_output (numpy.ndarray): The gradient of the loss function with respect to the output of this layer. This is computed in the subsequent layer during the backpropagation algorithm.

        Returns:
        - numpy.ndarray: The gradient of the loss function with respect to the input of this layer. This will be passed to the previous layer during the backpropagation algorithm.
        """
        relu_grad = self.last_input > 0
        return grad_output * relu_grad


class SoftmaxLayer:
    """
    This class represents a Softmax Layer in a neural network. It's generally used in the output layer of a
    multi-class classification problem. The softmax function takes a vector of arbitrary real-valued scores
    and squashes it to a vector of values between zero and one that sum to one.

    Attributes:
    ----------
    last_input_shape : tuple
        The shape of the last batch of input data passed through the layer. This is used during backpropagation
        to reshape gradients to their original shape.
    """

    def __init__(self):
        """
        Initializes the Softmax Layer. The attribute 'last_input_shape' is initialized as None.
        """
        self.last_input_shape = None

    def forward(self, input):
        """
        Perform a forward pass through the Softmax Layer. This function calculates the softmax of the input,
        which is a vector of raw predictions from a classification model.

        Parameters:
        ----------
        input : np.ndarray
            The raw scores outputted by the model.

        Returns:
        -------
        np.ndarray
            The softmax output, which represents the probabilities of each class for each instance in the input.
        """
        exp = np.exp(input)
        self.last_input_shape = input.shape
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def backward(self, grad_output):
        """
        Perform a backward pass through the Softmax Layer. This function calculates the gradient of the loss
        with respect to the input of the softmax function.

        Parameters:
        ----------
        grad_output : np.ndarray
            The gradient of the loss with respect to the output of the softmax function.

        Returns:
        -------
        np.ndarray
            The gradient of the loss with respect to the input of the softmax function.
        """
        exp = np.exp(input)
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)
        return grad_output * softmax * (1 - softmax)


class CNN:
    """
    This class represents a Convolutional Neural Network (CNN), a deep learning model most commonly applied in image
    recognition and processing. The CNN comprises one or more convolution layers, followed by activation functions,
    and concluded by a softmax layer for classification. Each layer transforms the input data, extracting high-level
    features for classification. In CNN architecture, the network learns these filters/features, rather than hand-crafting them.

    Attributes:
    ----------
    layers : list
        A list that stores the layers of the CNN. It can contain convolutional layers, pooling layers,
        fully connected layers and activation function layers.

    softmax : SoftmaxLayer
        A layer that implements the softmax function to be used in the output layer of the network.
        It converts the network's output into a probability distribution over predicted output classes.
    """

    def __init__(self):
        """
        Initializes the CNN with an empty list of layers and a softmax layer.
        """
        self.layers = []
        self.softmax = SoftmaxLayer()

    def add_layer(self, layer):
        """
        Add a new layer to the CNN.

        Parameters:
        ----------
        layer : object
            The layer object to be added to the network. This could be a Convolutional Layer,
            Fully Connected Layer, Max Pool Layer or any other type of layer.
        """
        self.layers.append(layer)

    def forward(self, X):
        """
        Perform a forward pass through all the layers of the CNN and return the softmax output.

        Parameters:
        ----------
        X : np.ndarray
            The input data for the CNN.

        Returns:
        -------
        np.ndarray
            The softmax output from the CNN, representing probabilities of each class for the input data.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return self.softmax.forward(X)

    def compute_loss_and_gradients(self, X, y):
        """
        Compute the loss and gradients for the CNN, given input data and true labels.

        Parameters:
        ----------
        X : np.ndarray
            The input data for the CNN.
        y : np.ndarray
            The true labels corresponding to X.

        Returns:
        -------
        tuple
            The computed loss and gradient output.
        """
        logits = self.forward(X)
        loss = -np.sum(y * np.log(logits))
        grad_output = logits
        grad_output = grad_output - y
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return loss, grad_output

    def train_step(self, X, y):
        """
        Perform a single training step, computing the loss and applying the gradients.

        Parameters:
        ----------
        X : np.ndarray
            The input data for the CNN.
        y : np.ndarray
            The true labels corresponding to X.

        Returns:
        -------
        float
            The computed loss for the current training step.
        """
        loss, grad_output = self.compute_loss_and_gradients(X, y)
        return loss

    def fit(self, X_train, y_train, epochs, X_val, y_val):
        """
        Train the CNN for a specified number of epochs and validate it.

        Parameters:
        ----------
        X_train : np.ndarray
            The training data for the CNN.
        y_train : np.ndarray
            The true labels corresponding to X_train.
        epochs : int
            The number of times the learning algorithm will work through the entire training dataset.
        X_val : np.ndarray
            The validation data for the CNN.
        y_val : np.ndarray
            The true labels corresponding to X_val.
        """
        for epoch in range(epochs):
            print("--- Epoch %d ---" % (epoch + 1))
            permutation = np.random.permutation(len(X_train))
            loss = 0
            for i in permutation:
                loss += self.train_step(X_train[i], y_train[i])
            print("Train loss:", loss / len(X_train))
            print(
                "Val loss:",
                np.mean(
                    [
                        self.compute_loss_and_gradients(X_val[i], y_val[i])[0]
                        for i in range(len(X_val))
                    ]
                ),
            )

    def predict(self, X):
        """
        Perform a forward pass through the network and return the predicted class for each input instance.

        Parameters:
        ----------
        X : np.ndarray
            The input data for which to make predictions.

        Returns:
        -------
        int
            The predicted class for each input instance.
        """
        return np.argmax(self.forward(X))

    def evaluate(self, X, y):
        """
        Compute the accuracy of the CNN's predictions, given input data and true labels.

        Parameters:
        ----------
        X : np.ndarray
            The input data for which to compute accuracy.
        y : np.ndarray
            The true labels corresponding to X.

        Returns:
        -------
        float
            The computed accuracy of the CNN's predictions.
        """
        return np.mean([self.predict(X[i]) == y[i] for i in range(len(X))])


# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = (X_train.astype(np.float32) / 255).reshape(-1, 28, 28, 1)
X_test = (X_test.astype(np.float32) / 255).reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Initialize CNN
cnn = CNN()

cnn.add_layer(
    ConvLayer(num_filters=32, filter_size=3, num_input_channels=1)
)  # Change num_input_channels to 3 for RGB images
cnn.add_layer(ReLULayer())
cnn.add_layer(MaxPoolLayer(pool_size=2))
cnn.add_layer(
    ConvLayer(num_filters=64, filter_size=3, num_input_channels=32)
)  # num_input_channels should be equal to the number of filters in the previous ConvLayer
cnn.add_layer(ReLULayer())
cnn.add_layer(MaxPoolLayer(pool_size=2))
cnn.add_layer(FullyConnectedLayer(output_dim=10))
cnn.add_layer(ReLULayer())

# Train the model
cnn.fit(X_train, y_train, epochs=10, X_val=X_test, y_val=y_test)
