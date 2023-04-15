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
        :return:
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
        :param epochs: the number of epochs to train the perceptron for,
                        if 0 the perceptron will train until it converges
        :return:
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
