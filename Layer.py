import numpy as np

import Neuron

class Layer:
    def __init__(self, inputSize, size, hard, activation = Neuron.Sigmoid()):
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

        sigmoid = Neuron.ActivationFunction()
        for i in range(0, self.size):
            self.neurons.append(Neuron.Neuron(sigmoid))

        # Random initial weights and biases
        # np.random.seed(0)
        # self.weights = np.random.rand(self.inputSize, self.size)
        if hard == True:
            self.weights = np.array([[0.15, 0.25],[0.2, 0.30]]).T
            self.biases = np.array([[0.35, 0.35]])
        else:
            self.weights = np.array([[0.40, 0.50],[0.45, 0.55]]).T
            self.biases = np.array([[0.60, 0.60]])
        # np.random.seed(0)
        # self.biases = np.random.rand(1, self.size)

    def prepareNewWeights(self, newWeights):
        self.newWeights = newWeights

    def applyNewWeights(self):
        # Internally here, the weights are not transposed
        # print("Old weights")
        # print(self.weights.T)
        self.weights = self.newWeights.T

        # print("New weights")
        # print(self.newWeights)

    def getInputSize(self):
        return self.inputSize

    def getLayerSize(self):
        return self.size

    # Return copy to not override the weights
    def getWeightsCopy(self):
        return self.weights.copy()


    def forwardPropagation(self, inputData):
        print("Input^T:", inputData.T.shape)
        print(inputData.T)
        print()
        # print("Input^T shape:", inputData.T.shape)
        print("Weights^T: ", self.weights.T.shape)
        print(self.weights.T)
        print()
        print("Biases:", self.biases.shape)
        print(self.biases)
        print()
        # print("Rs", np.dot(self.weights.T, inputData.T) + self.biases.T)
        # np.dot(self.weights.T, inputData.T) + self.biases.T) 
        # and np.dot(inputData, self.weights) + self.biases are the same
        self.rawOutput = np.dot(inputData, self.weights.T) + self.biases
        print("Ouput before activation:", self.rawOutput.shape)
        print(self.rawOutput)
        print()
        self.output = self.activation.activate(self.rawOutput)

    def getOutput(self):
        return self.output