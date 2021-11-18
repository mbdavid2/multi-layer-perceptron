import numpy as np

class ActivationFunction:
    def activate(self, inputValue):
        return 0*inputValue


class Sigmoid(ActivationFunction):
    def activate(self, inputValue):
        if inputValue > 0.5:
            return 1
        else:
            return 0

class Same(ActivationFunction):
    def activate(self, inputValue):
        return inputValue


class Neuron:
    def __init__(self, activationF):
        self.activationF = activationF

    def activate(self, input):
        return self.activationF.activate(input)
    

class Layer:
    def __init__(self, inputSize, size, activation = Same()):
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
        print("Input size: ", len(inputData))
        self.output = np.dot(self.weights.T, inputData) + self.biases

    def getOutput(self):
        return [self.activation.activate(sum(x)) for x in self.output]


class Network:
    def __init__(self, layers):
        self.layers = layers

    def feedForward(self, inputData):
        # Check that the input size is the same as layers[0]
        if len(inputData) != self.layers[0].getInputSize():
            errorStr = ('Input size: ' + str(len(inputData)) + 
                       ' != first layer size: ' + str(self.layers[0].getInputSize()))
            raise ValueError(errorStr)

        nextInput = inputData
        i = 0
        for layer in self.layers:
            print('Layer', i)
            layer.forwardPropagation(nextInput)
            nextInput = layer.getOutput()
            print("Output of the layer: ", nextInput)
            i = i + 1