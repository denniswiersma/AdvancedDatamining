class Perceptron:
    """
    Implementation of a perceptron, which is a single layer neural network.
    """

    def __init__(self, dimensions: int):
        """
        Initializes the perceptron.
        :param dimensions: number of dimensions of the perceptron, otherwise known as the number of weights or inputs
        """
        self.dimensions: int = dimensions
        self.bias: float = 0.0  # otherwise known as w0
        self.weights: list[float] = [0.0] * dimensions  # every input has a weight

    def __repr__(self):
        """
        Returns a string representation of the perceptron.
        :return: string representation of the perceptron
        """
        return f"Perceptron(dimensions={self.dimensions})"

    def predict(self, inputs: list[list[float]]):
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
            pre_activation: float = sum([value * weight for value, weight in zip(input, self.weights)]) + self.bias
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
                self.weights = [weight - error * value for weight, value in zip(self.weights, input)]

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
