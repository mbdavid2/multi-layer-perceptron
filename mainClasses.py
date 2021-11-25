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
    

class Layer:
    def __init__(self, inputSize, size, activation = Sigmoid()):
        """dasdasda

        Keyword arguments:
        inputSize -- expected input size
        size -- number of neurons
        activationFunc -- activation function
        """
        self.inputSize = inputSize
        self.size = size
        self.neurons = []
        self.activation = activation

        sigmoid = ActivationFunction()
        for i in range(0, self.size):
            self.neurons.append(Neuron(sigmoid))

        # Random initial weights and biases
        np.random.seed(0)
        self.weights = np.random.rand(self.inputSize, self.size)
        np.random.seed(0)
        self.biases = np.random.rand(1, self.size)


    def getInputSize(self):
        return self.inputSize

    def getLayerSize(self):
        return self.size


    def forwardPropagation(self, inputData):
        print("Weights shape: ", self.weights.shape)
        print("Weights:", self.weights.T)
        print("Biases:", self.biases, self.biases.shape)
        print("Input^T: ", inputData.T, inputData.T.shape)
        # print("Rs", np.dot(self.weights.T, inputData.T) + self.biases.T)
        # np.dot(self.weights.T, inputData.T) + self.biases.T) 
        # and np.dot(inputData, self.weights) + self.biases are the same
        self.output = np.dot(inputData, self.weights) + self.biases

    def getOutput(self):
        print("Result:", self.output)
        activatedOutput = self.activation.activate(self.output)
        print("Activated result:", activatedOutput)
        return activatedOutput
        normalList = []
        for x in activatedOutput:
            for i in x:
             normalList.append(i)
        return normalList


class Network:
    def __init__(self, layers):
        self.layers = layers

    def feedForward(self, inputData):
        # Check that the input size is the same as layers[0]
        if inputData.shape[1] != self.layers[0].getInputSize():
            errorStr = ('Input size: ' + str(len(inputData)) + 
                       ' != first layer size: ' + str(self.layers[0].getInputSize()))
            raise ValueError(errorStr)

        nextInput = inputData
        i = 0
        for layer in self.layers:
            print("--------- Layer", i, "-------------")
            layer.forwardPropagation(nextInput)
            nextInput = layer.getOutput()
            print("Output of the layer: ", nextInput)
            i = i + 1