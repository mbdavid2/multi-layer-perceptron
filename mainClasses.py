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
        print("Input^T: ", inputData.T, inputData.T.shape)
        print("Input^T shape:", inputData.T.shape)
        print("Weights shape: ", self.weights.shape)
        print("Weights^T:", self.weights.T)
        print("Biases:", self.biases, self.biases.shape)
        # print("Rs", np.dot(self.weights.T, inputData.T) + self.biases.T)
        # np.dot(self.weights.T, inputData.T) + self.biases.T) 
        # and np.dot(inputData, self.weights) + self.biases are the same
        self.output = np.dot(inputData, self.weights) + self.biases

    def getOutput(self):
        print("Result:", self.output, self.output.shape)
        activatedOutput = self.activation.activate(self.output)
        return activatedOutput


class Network:
    def __init__(self, layers):
        self.layers = layers

    def checkDimensions(self, inputData, layer, layerN):
        # Check that the input size is the same as layers[0]
        if inputData.shape[1] != layer.getInputSize():
            errorStr = ('Stopped at Layer(' + str(layerN) + ') -> Incoming input size: ' 
                        + str(inputData.shape[1]) + ' != ' + str(layer.getInputSize()) + 
                       ' (Layer(' + str(layerN) + ') expected input size)')
            raise ValueError(errorStr)

    def softMax(self, outputValues):
        expValues = [np.e**x for x in outputValues[0]]
        total = sum(expValues)
        normValues = [x/total for x in expValues]
        return normValues


    def feedForward(self, inputData):
        nextInput = inputData
        for i, layer in enumerate(self.layers):
            self.checkDimensions(nextInput, layer, i)
            print("--------- Layer", i, "-------------")
            layer.forwardPropagation(nextInput)
            nextInput = layer.getOutput()
            print("Layer output (activated): ", nextInput, nextInput.shape)

        output = nextInput
        print("Output (Softmax):", self.softMax(output))