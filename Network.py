import numpy as np

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
        print('Feedforward output:', nextInput)
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


    # ∂C/∂a = (a - target)
    #       = 1/2(target - out)^2, power and 1/2 cancel each other out
    # ∂a/∂z = a*(1 - a) (derivative of sigmoid)
    #       f(x) = 1/(1+e^-x), then f'(x) = f(x)(1-f(x))
    # ∂z/∂w = a_prev (previous activated output)
    #       = w*a_prev (this one) + w*a_prev (other weights) + b = a_prev + 0 + 0

    def computeWeightsOutputLayer(self, outputPrev, output, target, layer, learningRate = 0.25):
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

        print("New weights output layer:")
        print(weights)
        layer.prepareNewWeights(weights)
        return partialDeltas


    def computeWeights(self, outputPrev, output, layer, nextLayer, partialDeltas, learningRate = 0.25):
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
        print("New weights:")
        print(weights)
        layer.prepareNewWeights(weights)
        return newPartialDeltas

    
    def train(self, inputData, target):
        if (len(inputData) != len(target)):
            errorStr = 'Input and output differ in size'
            raise ValueError(errorStr)
        
        # Propagate the input forward
        allOutputs = self.feedForward(inputData)
        
        print('Target:', target)
        totalCost = 0
        for j in range(0, len(inputData)):
            squaredError = self.computeError(allOutputs[-1][j], target[j])
            totalCost = totalCost + squaredError
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
                    partialDeltas = self.computeWeights(outputPrev, output, layer, self.layers[i+1], partialDeltas)

        print('Total cost:', totalCost)

        # Apply the new weights
        print('Applying new weights...')
        for layer in self.layers:
            layer.applyNewWeights()
        
        print('Running feedforward again...')
        allOutputs = self.feedForward(inputData)

        # Compute new error
        totalCost = 0
        for j in range(0, len(inputData)):
            squaredError = self.computeError(allOutputs[-1][j], target[j])
            totalCost = totalCost + squaredError
        print('New total cost:', totalCost)


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

        
