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
    def __init__(self, inputSize, size, hard, activation = Sigmoid()):
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
        # np.random.seed(0)
        # self.weights = np.random.rand(self.inputSize, self.size)
        if hard == True:
            self.weights = np.array([[0.15, 0.2],[0.25, 0.30]]).T
            self.biases = np.array([[0.35, 0.35]])
        else:
            self.weights = np.array([[0.40, 0.45],[0.50, 0.55]]).T
            self.biases = np.array([[0.60, 0.60]])
        # np.random.seed(0)

        # self.biases = np.random.rand(1, self.size)


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
        self.rawOutput = np.dot(inputData, self.weights) + self.biases
        print("Ouput before activation:", self.rawOutput.shape)
        print(self.rawOutput)
        print()
        self.output = self.activation.activate(self.rawOutput)

    def getOutput(self):
        return self.output


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.nLayers = len(layers)

    def checkDimensions(self, inputData, layer, layerN):
        # Check that the input size is the same as layers[0]
        if inputData.shape[1] != layer.getInputSize():
            errorStr = ('Stopped at Layer(' + str(layerN) + ') -> Incoming input size: ' 
                        + str(inputData.shape[1]) + ' != ' + str(layer.getInputSize()) + 
                       ' (Layer(' + str(layerN) + ') expected input size)')
            raise ValueError(errorStr)


    def softMax(self, outputValues):
        exponents = [np.e**x for x in outputValues]
        total = sum(exponents)
        normValues = [x/total for x in exponents]
        return normValues

    
    def softMaxBatch(self, outputValues):
        exponents = np.exp(outputValues)
        # print("exponents:", exponents)
        totalSums = np.sum(exponents, axis=1, keepdims=True)
        # print("TotalSums:", totalSums, totalSums.shape)
        normValues = exponents / totalSums
        # print("Norm values")
        # print(normValues)
        # normValues = [self.softMax(x) for x in outputValues]
        return normValues


    def feedForward(self, inputData):
        nextInput = inputData
        # Save input as the output of a first unexisting layer
        allOutputs = [inputData]
        for i, layer in enumerate(self.layers):
            self.checkDimensions(nextInput, layer, i)
            print()
            print("--------- Feedforward Layer", str(i) + "/" + str(self.nLayers-1), "-------------")
            layer.forwardPropagation(nextInput)
            nextInput = layer.getOutput()
            allOutputs.append(nextInput)
            print("Layer output (activated): ", nextInput.shape)
            print(nextInput)
            print()

        # output = nextInput
        return allOutputs
        # softMaxRes = self.softMaxBatch(output)
        # print("Output (Softmax):", softMaxRes.shape)
        # print(softMaxRes)

    def computeError(self, networkOutput, target):
        # SquaredErrors will be the array with the 
        # errors for each output neuron against the 
        # desired
        squaredErrors = (1/2)*np.power(target - networkOutput, 2)
        return sum(squaredErrors)


    # The activation one should depend on the activation function we're using!!!
    # ∂C/∂a = (a - target)
    #       = 1/2(target - out)^2, power and 1/2 cancel each other out
    # ∂a/∂z = a*(1 - a) (derivative of sigmoid)
    #       f(x) = 1/(1+e^-x), then f'(x) = f(x)(1-f(x))
    # ∂z/∂w = a_prev (previous activated output)
    #       = w*a_prev (this one) + w*a_prev (other weights) + b = a_prev + 0 + 0
    # def computeWeightsOutputLayer(self, outputPrev, output, target, learningRate = 0.5):
    #     # C = total error (C_0 + C_1 + ... C_n), a = activated output
    #     # z = raw output, w = weight
    #     # ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a
    #     print("Output:", output.shape[0])
    #     weights = self.layers[-1].getWeights().T
    #     eta = learningRate
    #     for i, neuron in enumerate(output):
    #         for j, neuronPrev in enumerate(outputPrev):
    #             # w' = w - eta*∂C/∂w
    #             delta = -(target[i] - neuron)*neuron*(1 - neuron)*neuronPrev
    #             weights[i, j] = weights[i, j] - eta*delta

    #     print(weights)
    #     return weights

    def computeWeightsOutputLayer(self, outputPrev, output, target, layer, learningRate = 0.5):
        # C = total error (C_0 + C_1 + ... C_n), a = activated output
        # z = raw output, w = weight
        # ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a
        print("Output:", output.shape[0])
        print(output)

        print("Prev. Output:", outputPrev.shape[0])
        print(outputPrev)

        weights = layer.getWeightsCopy().T
        partialDeltas = np.zeros(weights.shape)
        for i, neuron in enumerate(output):
            for j, neuronPrev in enumerate(outputPrev):
                # We need: ∂z/∂w*∂a/∂z*∂C/∂a. Store ∂a/∂z*∂C/∂a for later
                partialDeltas[i, j] = -(target[i] - neuron)*neuron*(1 - neuron)

                # w' = w - eta*∂C/∂w | ∂C/∂w = ∂z/∂w*(∂a/∂z*∂C/∂a), (∂a/∂z*∂C/∂a) = partialDelta
                weights[i, j] = weights[i, j] - learningRate*(neuronPrev*partialDeltas[i, j])

        print("Delta weights output layer:")
        print(weights)
        return partialDeltas


    def computeWeights(self, outputPrev, output, target, layer, nextLayer, partialDeltas, learningRate = 0.5):
        eta = learningRate
        weights = layer.getWeightsCopy().T
        nextWeightsRef = nextLayer.weights.T
        print("Output:", output.shape[0])
        print(output)

        print("Prev. Output:", outputPrev.shape[0])
        print(outputPrev)

        print("Weights")
        print(weights)

        newPartialDeltas = np.zeros(weights.shape)

        for i, neuron in enumerate(output):
            for j, neuronPrev in enumerate(outputPrev):
                # ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a
                # ∂C/∂a = "total" = ∂C/∂a = ∂C_0/∂a + ... + ∂C_n/∂a
                total = 0
                for t, neuron in enumerate(output):
                    # The neuron affects multiple neurons on the next layer
                    # ∂C_i/∂a = ∂C_i/∂z*∂z/∂a = a*b, index with t instead of i for the output
                    a = partialDeltas[t, j] # a = ∂C_i/∂z = ∂C_i/∂a*∂a/∂z, already computed
                    b = nextWeightsRef[t, j] # b = ∂z/∂a = w]
                    total = total + a*b
                # We need: ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a = ∂z/∂w*∂a/∂z*total. Store ∂a/∂z*total for later
                newPartialDeltas[i, j] = total*neuron*(1 - neuron)
                weights[i, j] = weights[i, j] - eta*newPartialDeltas[i, j]*neuronPrev
        print("Delta weights:", weights)
        return newPartialDeltas

    
    def train(self, inputData, target):
        if (len(inputData) != len(target)):
            errorStr = 'Input and output differ in size'
            raise ValueError(errorStr)
        
        # Propagate the input forward
        self.feedForward(inputData)
        
        # print('Feedforward output:', outputs[-1])
        print('Target:', target)

        for j in range(0, len(inputData)):
            for i in range(self.nLayers - 1, -1, -1):
                print()
                print("--------- Backpropagation Layer", str(i) + "/" + str(self.nLayers-1), "-------------")
                layer = self.layers[i]
                output = layer.getOutput()[j]

                if i == 0:
                    outputPrev = inputData[j]
                else:
                    layerPrev = self.layers[i-1]
                    outputPrev = layerPrev.getOutput()[j]

                if i == self.nLayers - 1:
                    partialDeltas = self.computeWeightsOutputLayer(outputPrev, output, target[j], layer)
                else:
                    partialDeltas = self.computeWeights(outputPrev, output, target[j], layer, self.layers[i+1], partialDeltas)




        # for i, x in enumerate(outputs):
        #     if i == len(outputs) - 1:
        #         self.computeWeightsOutputLayer(outputs[i-1][0], x[0], target, self.layers[i-1])
        #     elif i != 0:
        #         # print("now", x)
        #         # print("previous layer", outputs[i-1])
        #         # squaredError = self.computeError(x, target[i])
        #         # totalCost = totalCost + squaredError
        #         # self.computeWeights(outputs[-2][i], x, target[i])
        #         self.computeWeights(outputs[i-1][0], x[0], target, self.layers[i-1], self.layers[i])
        #     print()
        # ## totalCost should be averaged???

        # print('Total cost:', totalCost)
