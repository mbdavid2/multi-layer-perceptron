import numpy as np

class ActivationFunction:
    def activate(self, inputValue):
        return inputValue

    def derivative(self, inputValue):
        return inputValue


class Sigmoid(ActivationFunction):
    def activate(self, inputValue):
        # print(inputValue)
        # if inputValue < 0:
        #     return np.exp(inputValue)/(1 + np.exp(inputValue))
        # else:
        return 1 / (1 + np.exp(-inputValue))

    def derivative(self, inputValue):
        return inputValue*(1 - inputValue)


class Same(ActivationFunction):
    def activate(self, inputValue):
        return inputValue

    def derivative(self, inputValue):
        return 1


class Tanh(ActivationFunction):
    def activate(self, inputValue):
        return np.tanh(inputValue)


class Neuron:
    def __init__(self, activationF):
        self.activationF = activationF

    def activate(self, input):
        return self.activationF.activate(input)