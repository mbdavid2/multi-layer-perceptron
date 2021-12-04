import numpy as np

class ActivationFunction:
    def activate(self, inputValue):
        return 0*inputValue


class Sigmoid(ActivationFunction):
    def activate(self, inputValue):
        return 1 / (1 + np.exp(-inputValue))


class Same(ActivationFunction):
    def activate(self, inputValue):
        return inputValue


class Neuron:
    def __init__(self, activationF):
        self.activationF = activationF

    def activate(self, input):
        return self.activationF.activate(input)