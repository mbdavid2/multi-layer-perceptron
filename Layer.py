import numpy as np

import Neuron

class Layer:
    def __init__(self, inputSize, size, smaller=False, activation = Neuron.Sigmoid()):
        """Layer constructor, creates the appropriate weight
        and bias matrices based on the layer size and expected
        input size

        Keyword arguments:
        inputSize -- expected input size
        size -- number of neurons of the layer
        activationFunc -- activation function (Sigmoid by default)
        """
        self.inputSize = inputSize
        self.size = size
        self.neurons = []
        self.activation = activation
        if smaller:
            self.weightFactor = 0.1
        else:
            self.weightFactor = 1

        ### TODO: The neuron instances are not really used, the act. function 
        # is simply called in the feedforward method

        sigmoid = Neuron.ActivationFunction()
        for i in range(0, self.size):
            self.neurons.append(Neuron.Neuron(sigmoid))

        self.setRandomWeights()
        
    def getActivationName(self, simple=False):
        if simple:
            return self.activation.nameShort()
        else:
            return self.activation.name()

    def setRandomWeights(self):
        # Random initial weights and biases
        np.random.seed(0)
        self.weights = np.random.rand(self.inputSize, self.size)*np.sqrt(1/self.inputSize)*self.weightFactor
        np.random.seed(0)
        self.biases = np.random.rand(1, self.size)*np.sqrt(1/self.inputSize)*self.weightFactor
        # self.biases = np.zeros(shape=(1, self.size))  + 0.1

    def reset(self):
        self.setRandomWeights()

    def activationDerivative(self, inputData):
        return self.activation.derivative(inputData)

    def prepareNewParameters(self, newWeights, newBiases):
        self.newWeights = newWeights
        self.newBiases = newBiases

    def applyNewParameters(self):
        self.weights = self.newWeights
        self.biases = self.newBiases

    def getInputSize(self):
        return self.inputSize

    def getLayerSize(self):
        return self.size

    # Return copy to not override the weights
    def getWeightsCopy(self):
        return self.weights.copy()

    def getBiasesCopy(self):
        return self.biases.copy()

    def getBiasesRef(self):
        return self.biases

    def getWeightsRef(self):
        return self.weights


    def forwardPropagation(self, inputData):
        self.rawOutput = np.dot(self.weights.T ,inputData.T) + self.biases.T

        # Do it directly applying the func to the whole array
        # rather than iterating over each neuron
        self.output = self.activation.activate(self.rawOutput).T

    def getOutput(self):
        return self.output