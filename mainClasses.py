import numpy as np

class ActivationFunction:
    def activate(self, input):
        if input > 0.5:
            return 1
        else:
            return 0


class Neuron:
    def __init__(self, activationF):
        self.activationF = activationF

    def activate(self, input):
        return self.activationF.activate(input)
    

class Layer:
    def __init__(self, inputSize, neurons):
        """dasdasda

        Keyword arguments:
        inputSize -- expected input size
        neurons -- list of neurons (type neuron)
        """
        self.inputSize = inputSize
        self.neuronNumber = len(neurons)

        # Random initial weights and biases
        np.random.seed(0)
        self.weights = np.random.rand(self.inputSize, self.neuronNumber)
        np.random.seed(0)
        self.biases = np.random.rand(1, self.neuronNumber)

    def forwardPropagation(self, inputData):
        self.output = np.dot(inputData, np.array(self.weights).T) + self.biases

    def getOutput(self):
        return self.output