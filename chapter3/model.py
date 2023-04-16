import math
from collections import Counter
from copy import deepcopy
from typing import Callable


class Perceptron:
    """
    Implementation of a perceptron, which is a single layer neural network.
    """

    def __init__(self, dimensions: int) -> None:
        """
        Initializes the perceptron.
        :param dimensions: number of dimensions of the perceptron, otherwise known as the number of weights or inputs
        """
        self.dimensions: int = dimensions
        self.bias: float = 0.0  # otherwise known as w0
        self.weights: list[float] = [0.0] * dimensions  # every input has a weight

    def __repr__(self) -> str:
        """
        Returns a string representation of the perceptron.
        :return: string representation of the perceptron
        """
        return f"Perceptron(dimensions={self.dimensions})"

    def predict(self, inputs: list[list[float]]) -> list[int]:
        """
        Predicts the output of the perceptron for a given input.
        :param inputs: a list of inputs containing a list of values for each input
        :return: a list of predictions for each input
        """
        # Initialize a list to store the predictions
        predictions: list[int] = []
        # loop through list of inputs
        for input in inputs:
            # calculate the pre-activation value for every input
            pre_activation: float = (
                sum([value * weight for value, weight in zip(input, self.weights)])
                + self.bias
            )
            # apply the activation function (signum) to the pre-activation value
            if pre_activation > 0:
                predictions.append(1)
            elif pre_activation < 0:
                predictions.append(-1)
            else:
                predictions.append(0)

        return predictions

    def partial_fit(self, inputs, targets) -> None:
        """
        Partially fits the perceptron to the given inputs and targets.
        :param inputs: a list of inputs containing a list of values for each input
        :param targets: a list of target values for each input
        """
        # loop through the inputs and targets
        for input, target in zip(inputs, targets):
            # predict the output for the given input
            prediction: int = self.predict([input])[0]
            # calculate the error
            error: int = prediction - target
            # update the bias and weights if the error is not 0
            if error != 0.0:
                self.bias -= error
                self.weights = [
                    weight - error * value for weight, value in zip(self.weights, input)
                ]

    def fit(self, inputs, targets, *, epochs: int = 0) -> None:
        """
        Fully fits the perceptron to the given inputs and targets.
        :param inputs: a list of inputs containing a list of values for each input
        :param targets: a list of target values for each input
        :param epochs: the number of epochs to train the perceptron for,
                        if 0 the perceptron will train until it converges
        """
        # if epochs is 0, train until convergence
        if epochs == 0:
            # initialize the number of performed epochs
            performed_epochs: int = 0
            # initialize a boolean to check if the perceptron has converged
            finished: bool = False

            # loop until the perceptron has converged
            while not finished:
                # store the previous bias and weights
                previous_bias: float = self.bias
                previous_weights: list[float] = self.weights

                # train the perceptron for one epoch
                self.partial_fit(inputs, targets)
                # increment the number of performed epochs
                performed_epochs += 1

                # check if the perceptron has converged
                if previous_bias == self.bias and previous_weights == self.weights:
                    # set the finished boolean to True
                    finished = True
                    # print the number of performed epochs
                    print(f"Finished after {performed_epochs} epochs.")
        else:
            # loop through the given number of epochs
            for _ in range(epochs):
                # train the perceptron for one epoch
                self.partial_fit(inputs, targets)
                # print the number of performed epochs
                print(f"Finished after {epochs} epochs.")


class LinearRegression:
    """
    Implementation of a perceptron capable of doing linear regression.
    """

    def __init__(self, dimensions: int) -> None:
        """
        Initializes the perceptron.
        :param dimensions: number of dimensions of the perceptron, otherwise known as the number of weights or inputs
        """
        self.dimensions: int = dimensions
        self.bias: float = 0.0  # otherwise known as w0
        self.weights: list[float] = [0.0] * dimensions  # every input has a weight

    def __repr__(self) -> str:
        """
        Returns a string representation of the perceptron.
        :return: string representation of the perceptron
        """
        return f"LinearRegression(dimensions={self.dimensions})"

    def predict(self, inputs: list[list[float]]) -> list[float]:
        """
        Predicts the output of the perceptron for a given input.
        :param inputs: a list of inputs containing a list of values for each input
        :return: a list of predictions for each input
        """
        # Initialize a list to store the predictions
        predictions: list[float] = []
        # loop through list of inputs
        for input in inputs:
            # calculate the pre-activation value for every input
            pre_activation: float = (
                sum([value * weight for value, weight in zip(input, self.weights)])
                + self.bias
            )
            # apply the activation function (identity) to the pre-activation value
            post_activation: float = pre_activation
            # append the prediction to the list of predictions
            predictions.append(post_activation)

        return predictions

    def partial_fit(self, inputs, targets, *, alpha=0.01) -> None:
        """
        Partially fits the perceptron to the given inputs and targets.
        :param alpha: the learning rate
        :param inputs: a list of inputs containing a list of values for each input
        :param targets: a list of target values for each input
        """
        # loop through the inputs and targets
        for input, target in zip(inputs, targets):
            # predict the output for the given input
            prediction: float = self.predict([input])[0]
            # calculate the error
            error: float = prediction - target
            # update the bias and weights if the error is not 0
            if error != 0.0:
                self.bias -= alpha * error
                self.weights = [
                    weight - (alpha * error) * value
                    for weight, value in zip(self.weights, input)
                ]

    def fit(self, inputs, targets, *, alpha=0.01, epochs: int = 100) -> None:
        """
        Fully fits the perceptron to the given inputs and targets.
        :param alpha: the learning rate
        :param inputs: a list of inputs containing a list of values for each input
        :param targets: a list of target values for each input
        :param epochs: the number of epochs to train the perceptron for
        """
        # loop through the given number of epochs
        for _ in range(epochs):
            # train the perceptron for one epoch
            self.partial_fit(inputs, targets, alpha=alpha)
            # print the number of performed epochs
        print(f"Finished after {epochs} epochs.")


### Activation functions ###
def linear(pre_activation: float) -> float:
    """
    Applies the linear activation function to the given pre-activation value.
    :param pre_activation: the pre-activation value
    :return: the post-activation value
    """
    return pre_activation


def sign(pre_activation: float) -> int:
    """
    Applies the signum activation function to the given pre-activation value.
    :param pre_activation: the pre-activation value
    :return: the post-activation value
    """
    if pre_activation > 0:
        return 1
    elif pre_activation < 0:
        return -1
    else:
        return 0


def tanh(pre_activation: float) -> float:
    """
    Applies the hyperbolic tangent activation function to the given pre-activation value.
    :param pre_activation: the pre-activation value
    :return: the post-activation value
    """
    return math.tanh(pre_activation)


### Loss functions ###
def mean_squared_error(prediction: float, target: float):
    """
    Calculates the mean squared error between the target and the prediction.
    :param target: the target value
    :param prediction: the prediction value
    :return: the mean squared error
    """
    return (prediction - target) ** 2


def mean_absolute_error(prediction: float, target: float):
    """
    Calculates the mean absolute error between the target and the prediction.
    :param target: the target value
    :param prediction: the prediction value
    :return: the mean absolute error
    """
    return abs(prediction - target)


def hinge(prediction: float, target: float):
    """
    Calculates the hinge loss between the target and the prediction.
    :param target: the target value
    :param prediction: the prediction value
    :return: the hinge loss
    """
    return max(1 - (prediction * target), 0)


### Derivative function ###
def derivative(function: callable, delta: float = 0.01) -> callable:
    """
    Calculates the derivative of the given function.
    :param function: the function to calculate the derivative of
    :param delta: the delta to use for the derivative calculation
    :return: the derivative of the given function
    """

    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    # copy the name and qualname of the given function to the wrapper function
    wrapper_derivative.__name__ = function.__name__ + "’"
    wrapper_derivative.__qualname__ = function.__qualname__ + "’"

    return wrapper_derivative


class Neuron:
    """
    Implementation of a single neuron.
    """

    def __init__(
        self,
        dimensions: int,
        activation: Callable[[float], float] = linear,
        loss: Callable[[float, float], float] = mean_squared_error,
    ) -> None:
        """
        Initializes the neuron.
        :param dimensions: number of dimensions of the neuron, otherwise known as the number of weights or inputs
        :param activation: the activation function
        :param loss: the loss function
        """
        self.dimensions: int = dimensions
        self.bias: float = 0.0  # otherwise known as w0
        self.weights: list[float] = [0.0] * dimensions  # every input has a weight
        self.activation: Callable[[float], float] = activation  # activation function
        self.loss: Callable[[float, float], float] = loss  # loss function

    def __repr__(self) -> str:
        """
        Returns a string representation of the neuron.
        :return: string representation of the neuron
        """
        return f"Neuron(dimensions={self.dimensions}, activation={self.activation.__name__}, loss={self.loss.__name__})"

    def predict(self, inputs: list[list[float]]) -> list[float]:
        """
        Predicts the output of the neuron for a given input.
        :param inputs: a list of inputs containing a list of values for each input
        :return: a list of predictions for each input
        """
        # Initialize a list to store the predictions
        predictions: list[float] = []
        # loop through list of inputs
        for input in inputs:
            # calculate the pre-activation value for every input
            pre_activation: float = (
                sum([value * weight for value, weight in zip(input, self.weights)])
                + self.bias
            )
            # apply the activation function to the pre-activation value
            post_activation: float = self.activation(pre_activation)
            # append the prediction to the list of predictions
            predictions.append(post_activation)

        return predictions

    def partial_fit(self, inputs, targets, *, alpha=0.01) -> None:
        """
        Partially fits the neuron to the given inputs and targets.
        :param alpha: the learning rate
        :param inputs: a list of inputs containing a list of values for each input
        :param targets: a list of target values for each input
        """
        predictions: list[float] = self.predict(inputs)

        # loop through the inputs, targets, and predictions
        for input, target, prediction in zip(inputs, targets, predictions):
            # calculate the pre-activation value for every input
            pre_activation: float = (
                sum([value * weight for value, weight in zip(input, self.weights)])
                + self.bias
            )

            # update the bias
            self.bias = self.bias - alpha * derivative(self.loss)(
                prediction, target
            ) * derivative(self.activation)(pre_activation)

            # update the weights
            self.weights = [
                weight
                - alpha
                * derivative(self.loss)(prediction, target)
                * derivative(self.activation)(pre_activation)
                * value
                for value, weight in zip(input, self.weights)
            ]

    def fit(self, inputs, targets, *, alpha=0.001, epochs: int = 100) -> None:
        """
        Fully fits the neuron to the given inputs and targets.
        :param alpha: the learning rate
        :param inputs: a list of inputs containing a list of values for each input
        :param targets: a list of target values for each input
        :param epochs: the number of epochs to train the neuron for
        """
        # loop through the given number of epochs
        for _ in range(epochs):
            # train the perceptron for one epoch
            self.partial_fit(inputs, targets, alpha=alpha)
            # print the number of performed epochs
        print(f"Finished after {epochs} epochs.")


class Layer:
    """
    Implementation of a layer of neurons.
    """

    # Counter to keep track of the number of layers of each type
    class_counter = Counter()

    def __init__(self, outputs: int, *, name: str = None, next: "Layer" = None) -> None:
        """
        Initializes the layer.
        :param outputs: the number of outputs of the layer, otherwise known as the number of neurons
        :param name: the name of the layer
        :param next: the next layer in the network
        """
        # increment the counter for the current layer type
        Layer.class_counter[type(self)] += 1
        # if no name is given, use the default name
        if name is None:
            name = f"{type(self).__name__}_{Layer.class_counter[type(self)]}"

        self.inputs: int = 0  # number of inputs of the layer
        self.outputs: int = outputs  # number of outputs of the layer
        self.name: str = name  # name of the layer
        self.next: "Layer" = next  # the next layer in the network

    def __repr__(self) -> str:
        """
        Returns a string representation of the layer.
        :return: string representation of the layer
        """
        text = f"Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})"
        # if the layer has a next layer, add it to the string representation
        if self.next is not None:
            text += " + " + repr(self.next)
        return text

    def __getitem__(self, index: int | str) -> "Layer":
        """
        Returns the layer at the given index.
        :param index: the index of the layer
        :return: the layer at the given index
        """
        # if the index is 0 or the name of the layer, return the layer
        if index == 0 or index == self.name:
            return self

        # if the index is an integer, return the layer at the given index
        if isinstance(index, int):
            if self.next is None:
                raise IndexError("Layer index out of range")
            return self.next[index - 1]

        # if the index is a string, return the layer with the given name
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]

        # if the index is neither an integer nor a string, raise an error
        raise TypeError(
            f"Layer indices must be integers or strings, not {type(index).__name__}"
        )

    def __add__(self, next: "Layer") -> "Layer":
        """
        Adds a layer to the network. This method is called when using the + operator.
        :param next: the next layer in the network
        :return: a copy of the current layer with the given layer added
        """
        # create a copy of the current layer
        result = deepcopy(self)
        # add the given layer to the copy
        result.add(deepcopy(next))
        return result

    def add(self, next: "Layer") -> None:
        """
        Adds a layer to the network.
        :param next: the next layer in the network
        """
        # if the layer has no next layer, add the given layer as the next layer
        if self.next is None:
            self.next = next
            # set the number of inputs of the next layer to the number of outputs of the current layer
            next.set_inputs(self.outputs)
        # if the layer has a next layer, add the given layer to the next layer
        else:
            self.next.add(next)

    def set_inputs(self, inputs: int) -> None:
        """
        Sets the number of inputs of the layer.
        :param inputs: the number of inputs of the layer
        """
        self.inputs = inputs