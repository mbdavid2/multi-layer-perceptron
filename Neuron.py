import numpy as np

class ActivationFunction:
    def activate(self, inputValue):
        return inputValue

    def derivative(self, inputValue):
        return inputValue

    def name(self):
        return "basic"

    def nameShort(self):
        return "basic"


class Sigmoid(ActivationFunction):
    def activate(self, inputValue):
        return 1 / (1 + np.exp(-inputValue))

    def derivative(self, inputValue):
        return inputValue*(1 - inputValue)
    
    def name(self):
        return "sigmoid"

    def nameShort(self):
        return "sig"


class Same(ActivationFunction):
    def activate(self, inputValue):
        return inputValue

    def derivative(self, inputValue):
        return 1
    
    def name(self):
        return "identity"

    def nameShort(self):
        return "id"


class Tanh(ActivationFunction):
    def activate(self, inputValue):
        return np.tanh(inputValue)


class Neuron:
    def __init__(self, activationF):
        self.activationF = activationF

    def activate(self, input):
        return self.activationF.activate(input)