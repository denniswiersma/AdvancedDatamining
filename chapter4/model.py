import math
import random
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
    class_counter: Counter = Counter()

    def __init__(
        self, outputs: int | None, *, name: str = None, next: "Layer" = None
    ) -> None:
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
        self.outputs: int | None = outputs  # number of outputs of the layer
        self.name: str = name  # name of the layer
        self.next: "Layer" = next  # the next layer in the network

    def __repr__(self) -> str:
        """
        Returns a string representation of the layer.
        :return: string representation of the layer
        """
        text: str = f"Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})"
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

    def __call__(
        self,
        layer_inputs: list[list[float]],
        targets: list[list[float]] = None,
        alpha: float = None,
    ) -> tuple[list[list[float]], list[float], list[list[[float]]]]:
        """
        Calls the layer.
        :param layer_inputs: the inputs of the layer
        :return: the outputs of the layer
        """
        raise NotImplementedError("Abstract __call__ method called")

    def __add__(self, next: "Layer") -> "Layer":
        """
        Adds a layer to the network. This method is called when using the + operator.
        :param next: the next layer in the network
        :return: a copy of the current layer with the given layer added
        """
        # create a copy of the current layer
        result: "Layer" = deepcopy(self)
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
        self.inputs: int = inputs


class InputLayer(Layer):
    """
    Implementation of an input layer.
    """

    def __repr__(self) -> str:
        """
        Returns a string representation of the input layer.
        :return: string representation of the input layer
        """
        text: str = f"InputLayer(outputs={self.outputs}, name={repr(self.name)})"
        # if the layer has a next layer, add it to the string representation
        if self.next is not None:
            text += " + " + repr(self.next)
        return text

    def __call__(
        self,
        layer_inputs: list[list[float]],
        targets: list[list[float]] = None,
        alpha: float = None,
    ) -> tuple[list[list[float]], list[float], list[list[float]]]:
        """
        Calls the input layer.
        :param layer_inputs: the inputs of the layer
        :param targets: the expected outcomes of the network
        :return: the outputs of the input layer
        """
        return self.next(layer_inputs, targets, alpha)

    def predict(self, layer_inputs: list[list[float]]) -> list[list[float]]:
        """
        Predicts the outputs of the input layer which, by definition, are the same as the inputs.
        :param layer_inputs: the inputs of the input layer
        :return: the outputs of the input layer
        """
        prediction, _, _ = self(layer_inputs)
        return prediction

    def evaluate(
        self, layer_inputs: list[list[float]], targets: list[list[float]]
    ) -> float:
        """
        Evaluates the input layer.
        :param layer_inputs: the inputs of the input layer
        :param targets: the expected outcomes of the network
        :return: the loss of the input layer
        """
        _, losses, _ = self(layer_inputs, targets)
        mean_loss = sum(losses) / len(losses)
        return mean_loss

    def partial_fit(self, inputs, targets, alpha=0.001) -> None:
        """
        Partially fits the input layer to the given inputs and targets.
        :param inputs: a list of inputs containing a list of values for each input
        :param targets: a list of target values for each input
        :param alpha: the learning rate
        """
        self(inputs, targets, alpha)

    def fit(self, inputs, targets, *, alpha=0.001, epochs: int = 100) -> None:
        """
        Fully fits the neuron to the given inputs and targets.
        :param alpha: the learning rate
        :param inputs: a list of inputs containing a list of values for each input
        :param targets: a list of target values for each input
        :param epochs: the number of epochs to train the neuron for
        """
        # loop through the given number of epochs
        for epoch in range(epochs):
            # train the perceptron for one epoch
            self.partial_fit(inputs, targets, alpha=alpha)

            # print the number of performed epochs
        print(f"Finished after {epochs} epochs.")

    def set_inputs(self, inputs: int) -> None:
        """
        Sets the number of inputs of the layer.
        :param inputs: the number of inputs of the layer
        """
        raise NotImplementedError(
            "Input layer cannot receive inputs from another layer"
        )


class DenseLayer(Layer):
    """
    Implementation of a densely connected layer of neurons.
    """

    def __init__(self, outputs: int, *, name: str = None, next: "Layer" = None) -> None:
        """
        Initializes the dense layer.
        :param outputs: the number of outputs of the layer, otherwise known as the number of neurons
        :param name: the name of the layer
        :param next: the next layer in the network
        """
        # call the constructor of the super class
        super().__init__(outputs, name=name, next=next)
        self.bias: list[float] = [0.0] * outputs  # biases of the neurons
        self.weights: list[list[float]] | None = None  # weights of the neurons

    def __repr__(self) -> str:
        """
        Returns a string representation of the dense layer.
        :return: string representation of the dense layer
        """
        text: str = f"DenseLayer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})"
        # if the layer has a next layer, add it to the string representation
        if self.next is not None:
            text += " + " + repr(self.next)
        return text

    def __call__(
        self,
        layer_inputs: list[list[float]],
        targets: list[list[float]] = None,
        alpha: float = None,
    ) -> tuple[list[list[float]], list[float], list[list[float]]]:
        """
        Calls the dense layer.
        :param layer_inputs: the inputs of the layer
        :param targets: the expected outcomes of the network
        :return: the outputs of the layer
        """
        # initialise the list of pre-activations of all neurons in the layer
        layer_pre_activations: list[list[float]] = []
        gradients: list[list[float]] | None = None
        # for each neuron in the layer
        for neuron_inputs in layer_inputs:
            # initialise the list of pre-activations of the neuron
            neuron_pre_activations: list[float] = []
            # for each output of the neuron
            for output in range(self.outputs):
                # calculate the pre-activation of the neuron
                pre_activation: float = (
                    sum(
                        neuron_inputs[input] * self.weights[output][input]
                        for input in range(self.inputs)
                    )
                    + self.bias[output]
                )
                # add the pre-activation to the list of pre-activations of the neuron
                neuron_pre_activations.append(pre_activation)
            # add the list of pre-activations of the neuron to the list of pre-activations of all neurons in the layer
            layer_pre_activations.append(neuron_pre_activations)

        # pass the pre-activations to the next layer
        predictions, losses, received_gradients = self.next(
            layer_pre_activations, targets=targets, alpha=alpha
        )

        # if alpha is not None
        if alpha is not None:
            # initialise the list of gradients of the layer
            gradients: list[list[float]] = []

            # for each neuron in the layer and its received gradients
            for neuron_inputs, received_gradient in zip(
                layer_inputs, received_gradients
            ):
                # calculate the gradient for each input of the neuron
                gradient: list[float] = [
                    sum(
                        self.weights[output][input] * received_gradient[output]
                        for output in range(self.outputs)
                    )
                    for input in range(self.inputs)
                ]

                # add the gradient to the list of gradients of the layer
                gradients.append(gradient)

                # update the weights and biases of the neurons
                for output in range(self.outputs):
                    # update the bias of the neuron
                    self.bias[output] -= (
                        alpha / len(layer_inputs)
                    ) * received_gradient[output]

                    # update the weights of the neurons
                    self.weights[output] = [
                        self.weights[output][input]
                        - (alpha / len(layer_inputs))
                        * received_gradient[output]
                        * neuron_inputs[input]
                        for input in range(self.inputs)
                    ]

        return predictions, losses, gradients

    def set_inputs(self, inputs: int) -> None:
        """
        Sets the number of inputs of the layer and initializes the weights.
        :param inputs: the number of inputs of the layer
        """
        self.inputs = inputs
        # calculate the xavier initialisation constant
        xavier: float = math.sqrt(6 / (inputs + self.outputs))
        # if the weights have not been initialised yet
        if self.weights is None:
            # initialise the weights
            self.weights = [
                [random.uniform(-xavier, xavier) for _ in range(inputs)]
                for _ in range(self.outputs)
            ]


class ActivationLayer(Layer):
    """
    Implementation of an activation layer.
    """

    def __init__(
        self,
        outputs: int,
        activation: Callable[[float], float] = linear,
        *,
        name: str = None,
        next: "Layer" = None,
    ) -> None:
        """
        Initializes the activation layer.
        :param activation: the activation function
        :param name: the name of the layer
        :param next: the next layer in the network
        """
        # call the constructor of the super class
        super().__init__(outputs=outputs, name=name, next=next)
        self.activation: Callable[[float], float] = activation

    def __repr__(self) -> str:
        """
        Returns a string representation of the activation layer.
        :return: string representation of the activation layer
        """
        text: str = (
            f"ActivationLayer(inputs={self.inputs}, outputs={self.outputs}, activation={self.activation.__name__},"
            f" name={repr(self.name)})"
        )
        # if the layer has a next layer, add it to the string representation
        if self.next is not None:
            text += " + " + repr(self.next)
        return text

    def __call__(
        self,
        layer_inputs: list[list[float]],
        targets: list[list[float]] = None,
        alpha: float = None,
    ) -> tuple[list[list[float]], list[float], list[list[[float]]] | None]:
        """
        Calls the activation layer.
        :param layer_inputs: the inputs of the layer
        :return: the outputs of the layer
        """
        # initialise the list of post-activations of all neurons in the layer
        layer_post_activations: list[list[float]] = []
        gradients: list[list[float]] | None = None
        # for each list of pre-activations for each neuron in the layer
        for neuron_pre_activations in layer_inputs:
            # initialise the list of post-activations of the neuron
            neuron_post_activations: list[float] = []
            # for each output of the neuron
            for output in range(self.outputs):
                # calculate the post-activation of the neuron
                post_activation: float = self.activation(neuron_pre_activations[output])
                # add the post-activation to the list of post-activations of the neuron
                neuron_post_activations.append(post_activation)
            # add the list of post-activations of the neuron to the list of post-activations of all neurons in the layer
            layer_post_activations.append(neuron_post_activations)

        # pass the activations to the next layer
        predictions, losses, received_gradients = self.next(
            layer_post_activations,
            targets=targets,
            alpha=alpha,
        )

        # if alpha is not None
        if alpha is not None:
            # initialise the list of gradients
            gradients: list[list[float]] = []

            # for each list of pre-activations for each neuron in the layer and each list of received gradients
            for neuron_pre_activation, received_gradient in zip(
                layer_inputs, received_gradients
            ):
                # calculate the gradient
                gradient: list[float] = [
                    derivative(self.activation)(neuron_pre_activation[output])
                    * received_gradient[output]
                    for output in range(self.outputs)
                ]

                # add the gradient to the list of gradients
                gradients.append(gradient)

        return predictions, losses, gradients


class LossLayer(Layer):
    """Implementation of a loss layer."""

    def __init__(
        self,
        loss: Callable[[float, float], float] = mean_squared_error,
        *,
        name: str = None,
    ) -> None:
        """
        Initializes the loss layer.
        :param loss: the loss function
        :param name: the name of the layer
        """
        # call the constructor of the super class
        super().__init__(outputs=None, name=name)
        self.loss: Callable[[float, float], float] = loss

    def __repr__(self) -> str:
        """
        Returns a string representation of the loss layer.
        :return: string representation of the loss layer
        """
        text: str = f"LossLayer(inputs={self.inputs}, loss={self.loss.__name__}, name={repr(self.name)})"
        # if the layer has a next layer, add it to the string representation
        if self.next is not None:
            text += " + " + repr(self.next)
        return text

    def __call__(
        self,
        layer_inputs: list[list[float]],
        targets: list[list[float]] = None,
        alpha: float = None,
    ) -> tuple[list[list[float]], list[float], list[list[float]]]:
        """
        Calls the loss layer.
        :param layer_inputs: the inputs of the layer which, in this case, are the predictions
        :param targets: the expected outcomes of the network
        :return: the outputs of the layer
        """
        predictions: list[list[float]] = layer_inputs
        losses: list[float] | None = None
        gradients: list[list[float]] | None = None

        if targets is not None:
            losses: list[float] = []
            # for each prediction and target
            for prediction, target in zip(predictions, targets):
                # calculate the loss
                loss: float = sum(
                    self.loss(prediction[output], target[output])
                    for output in range(self.inputs)  # self.inputs == self.outputs
                )
                # add the loss to the list of losses
                losses.append(loss)

            # if alpha is not None, calculate the gradient
            if alpha is not None:
                gradients: list[list[float]] = []
                for prediction, target in zip(predictions, targets):
                    gradient: list[float] = [
                        derivative(self.loss)(prediction[output], target[output])
                        for output in range(self.inputs)
                    ]
                    # add the gradient to the list of gradients
                    gradients.append(gradient)

        return predictions, losses, gradients

    def add(self, next: "Layer") -> None:
        """
        Adds a layer to the network.
        :param next: the next layer in the network
        """
        raise NotImplementedError("Loss layers cannot have a next layer.")
