import numpy as np

class Network:
    def __init__(self, layers, learningRate = 0.9):
        self.layers = layers
        self.nLayers = len(layers)
        self.learningRate = learningRate

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
            # if i == len(self.layers)-1:
            #     nextInput = self.softMax(nextInput)
            allOutputs.append(nextInput)
            # print("Layer output (activated): ", nextInput.shape)
            # print(nextInput)
            # print()

        # output = nextInput
        # print('Feedforward output:', nextInput)
        return allOutputs
        # softMaxRes = self.softMaxBatch(output)
        # print("Output (Softmax):", softMaxRes.shape)
        # print(softMaxRes)

    def computeError(self, networkOutput, target):
        # SquaredErrors will be the array with the errors 
        # for each output neuron against the desired
        # print("My output")
        # print(networkOutput)

        # print("Tarrget")
        # print(target)
        squaredErrors = (1/2)*np.power(target - networkOutput, 2)
        return sum(squaredErrors)


    # ∂C/∂a = (a - target)
    #       = 1/2(target - out)^2, power and 1/2 cancel each other out
    # ∂a/∂z = a*(1 - a) (derivative of sigmoid)
    #       f(x) = 1/(1+e^-x), then f'(x) = f(x)(1-f(x))
    # ∂z/∂w = a_prev (previous activated output)
    #       = w*a_prev (this one) + w*a_prev (other weights) + b = a_prev + 0 + 0
    def computeWeightsOutputLayer(self, outputPrev, output, target, layer):
        # C = total error (C_0 + C_1 + ... C_n), a = activated output
        # z = raw output, w = weight
        # ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a
        # print("Output:", output.shape[0])
        # print(output)

        # print("Prev. Output:", outputPrev.shape[0])
        # print(outputPrev)

        weights = layer.getWeightsCopy()
        partialDeltas = np.zeros(weights.shape)
        # print("Weights.T")
        # print(weights)
        for i, neuronPrev in enumerate(outputPrev):
            for j, neuron in enumerate(output):
                # print("-----")
                # print("Doing weight:", i, j, weights[i, j])
                # We need: ∂z/∂w*∂a/∂z*∂C/∂a. Store ∂a/∂z*∂C/∂a for later
                partialDeltas[i, j] = -(target[j] - neuron)*neuron*(1 - neuron)
                # print("-(target[j] - neuron)", -(target[j] - neuron))
                # print("neuron*(1 - neuron)", neuron*(1 - neuron))
                # print("neuronPrev", neuronPrev)
                # print("neuron", neuron)
                # print("target", target[j])
                # print(neuronPrev*partialDeltas[i, j], -(target[j] - neuron)*neuronPrev*neuron*(1 - neuron))
                # w' = w - eta*∂C/∂w | ∂C/∂w = ∂z/∂w*(∂a/∂z*∂C/∂a), (∂a/∂z*∂C/∂a) = partialDelta
                weights[i, j] = weights[i, j] - self.learningRate*(neuronPrev*partialDeltas[i, j])
        #         print("new weight:", weights[i, j])
        # print("New weights output layer:")
        # print(weights)
        # print("deltas")
        # print(partialDeltas)
        layer.prepareNewWeights(weights)
        return partialDeltas


    def computeWeights(self, outputPrev, output, layer, nextLayer, partialDeltas):
        weights = layer.getWeightsCopy()
        nextWeightsRef = nextLayer.weights
        # print("Output:", output.shape[0])
        # print(output)

        # print("Prev. Output:", outputPrev.shape[0])
        # print(outputPrev)

        # print("Weights")
        # print(weights)

        # print("given deltas")
        # print(partialDeltas.shape)

        # print("prev weights")
        # print(nextWeightsRef)

        newPartialDeltas = np.zeros(weights.shape)

        for i, neuronPrev in enumerate(outputPrev):
            for j, neuron in enumerate(output):
                # print("-----")
                # print("Doing weight:", i, j, weights[i, j])
                # print("neuronPrev", neuronPrev)
                # print("neuron", neuron)
                # print("target", target[j])
                # ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a
                # ∂C/∂a = "total" = ∂C/∂a = ∂C_0/∂a + ... + ∂C_n/∂a
                total = 0
                for t in range(0, partialDeltas.shape[1]):
                    # print("accessing", j, t)
                    # The neuron affects multiple neurons on the next layer
                    # ∂C_i/∂a = ∂C_i/∂z*∂z/∂a = a*b, index with t instead of i for the output
                    a = partialDeltas[j, t] # a = ∂C_i/∂z = ∂C_i/∂a*∂a/∂z, already computed
                    b = nextWeightsRef[j, t] # b = ∂z/∂a = w]
                    # print("accessing", j, t, "B:", b)
                    # a = 1
                    total = total + a*b
                # We need: ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a = ∂z/∂w*∂a/∂z*total. Store ∂a/∂z*total for later
                newPartialDeltas[i, j] = total*neuron*(1 - neuron)
                weights[i, j] = weights[i, j] - self.learningRate*newPartialDeltas[i, j]*neuronPrev
        # print("New weights:")
        # print(weights)
        layer.prepareNewWeights(weights)
        return newPartialDeltas

    def test(self, inputData, target):   
        print('Running feedforward...')
        allOutputs = self.feedForward(inputData)

        # Compute new error
        totalCost = 0
        for j in range(0, len(inputData)):
            squaredError = self.computeError(allOutputs[-1][j], target[j])
            totalCost = totalCost + squaredError
        
        print('Feedforward output:', allOutputs[-1])
        print('Total cost:', totalCost)
        return allOutputs[-1]

    
    def train(self, inputData, target):
        if (len(inputData) != len(target)):
            errorStr = 'Input and output differ in size'
            raise ValueError(errorStr)
        
        print('Target:', target)
        totalCost = 0

        # Propagate the input forward, get the outputs (also stored inside each layer)
        allOutputs = self.feedForward(inputData)
        print('Output softmax', allOutputs[-1])


        for j in range(0, len(inputData)):
            # Compute the squared error with the last (-1, output layer) for the input case we're treating (j)
            squaredError = self.computeError(allOutputs[-1][j], target[j])
            totalCost = totalCost + squaredError

            for i in range(self.nLayers - 1, -1, -1):
                # print()
                # print("--------- Backpropagation Layer", str(i) + "/" + str(self.nLayers-1), "-------------")
                layer = self.layers[i]
                output = layer.getOutput()[j]
                # print(layer.getOutput())
                # exit(0)

                if i == 0:
                    outputPrev = inputData[j]
                else:
                    layerPrev = self.layers[i-1]
                    outputPrev = layerPrev.getOutput()[j]

                if i == self.nLayers - 1:
                    partialDeltas = self.computeWeightsOutputLayer(outputPrev, output, target[j], layer)
                else:
                    partialDeltas = self.computeWeights(outputPrev, output, layer, self.layers[i+1], partialDeltas)
            
            # Apply the new weights for this input case
            # print('Applying new weights...')
            for layer in self.layers:
                layer.applyNewWeights()
            
            print('Cost for last input:', squaredError)

        print('Total cost:', totalCost)


        
