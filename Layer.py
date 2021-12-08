import numpy as np

import Neuron

class Layer:
    def __init__(self, inputSize, size, activation = Neuron.Sigmoid()):
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

        ### TODO: The neurons are not used at all, the function is simply called in feedforward

        sigmoid = Neuron.ActivationFunction()
        for i in range(0, self.size):
            self.neurons.append(Neuron.Neuron(sigmoid))

        self.setRandomWeights()
        


    def setRandomWeights(self):
        # Random initial weights and biases
        np.random.seed(0)
        self.weights = (np.random.rand(self.inputSize, self.size) - 0.5) *0.1
        # self.weights = np.random.rand(self.inputSize, self.size)
        np.random.seed(0)
        self.biases = (np.random.rand(1, self.size) - 0.5) *0.25 #np.zeros(size=(1, self.size)) #(np.random.rand(1, self.size) - 0.5) *0.25
        self.biases = self.biases - self.biases + 0.1
        # self.biases = np.random.rand(1, self.size) 

    def reset(self):
        self.setRandomWeights()

    def activationDerivative(self, inputData):
        return self.activation.derivative(inputData)

    def prepareNewWeights(self, newWeights):
        self.newWeights = newWeights

    def applyNewWeights(self):
        self.weights = self.newWeights

    def getInputSize(self):
        return self.inputSize

    def getLayerSize(self):
        return self.size

    # Return copy to not override the weights
    def getWeightsCopy(self):
        return self.weights.copy()


    def forwardPropagation(self, inputData):
        # print("Input^T:", inputData.T.shape)
        # print(inputData.T)
        # print()
        # print("Input^T shape:", inputData.T.shape)
        # print("Weights^T: ", self.weights.T.shape)
        # print(self.weights.T)
        # print()
        # print("Biases:", self.biases.shape)
        # print(self.biases)
        # print()
        # print("Rs", np.dot(self.weights.T, inputData.T) + self.biases.T)
        # np.dot(self.weights.T, inputData.T) + self.biases.T) 
        # and np.dot(inputData, self.weights) + self.biases are the same
        self.rawOutput = np.dot(self.weights.T ,inputData.T) + self.biases.T
        # print("Ouput before activation:", self.rawOutput.shape)
        # print(self.rawOutput)
        # print()

        # Do it directly applying the func to the whole array
        # rather than iterating over each neuron
        self.output = self.activation.activate(self.rawOutput).T

    def getOutput(self):
        return self.output