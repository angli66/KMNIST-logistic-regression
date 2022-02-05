from audioop import ratecv
import numpy as np
import data
import time
import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""


def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    lower_bound, upper_bound = -32, 32 # To avoid overflow in np.exp
    np.clip(a, lower_bound, upper_bound, out = a)
    return 1 / (1 + np.exp(-a))

def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    return np.exp(a) / np.sum(np.exp(a), axis = 1).reshape(-1, 1)

def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = -(t*ln(y) + (1-t)*ln(1-y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    return -(t * np.log(y) + (1 - t) * np.log(1 - y))

def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    return -np.sum(t * np.log(y))

class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss
        self.weights = np.zeros((28*28+1, out_dim))

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for

        Returns
        _______
        n x (out_dim) matrix containing predicted output for every sample
        """
        return self.activation(X @ self.weights)

    def __call__(self, X):
        return self.forward(X)

    def train(self, minibatch):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        X, y = minibatch
        
        y = y.reshape(len(y), -1) # Reshape y to n x (out_dim)
        output = self(X) # n x (out_dim)

        # Update weights
        self.weights = self.weights - self.hyperparameters.learning_rate * (X.T @ (output - y))

        # Calculate loss and accuracy
        loss = np.sum(self.loss(output, y)) / len(X)
        if self.activation == sigmoid: # Logistic Regression
            accuracy = np.sum((np.absolute(output - y) < 0.5).astype(int)) / len(X)
        else: # Softmax regression
            prediction = data.onehot_decode(output)
            y_decode = data.onehot_decode(y)
            accuracy = np.sum((prediction == y_decode).astype(int)) / len(X)

        return loss, accuracy

    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """
        X, y = minibatch
        
        y = y.reshape(len(y), -1) # Reshape y to n x (out_dim)
        output = self(X) # n x (out_dim)

        # Calculate loss and accuracy
        loss = np.sum(self.loss(output, y)) / len(X)
        if self.activation == sigmoid: # Logistic Regression
            accuracy = np.sum((np.absolute(output - y) < 0.5).astype(int)) / len(X)
        else: # Softmax regression
            prediction = data.onehot_decode(output)
            y_decode = data.onehot_decode(y)
            accuracy = np.sum((prediction == y_decode).astype(int)) / len(X)

        return loss, accuracy