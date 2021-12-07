import numpy as np
import logging

class Network:
    def __init__(self, layers, learningRate = 0.001):
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


    def reset(self):
        for layer in self.layers:
            layer.reset()

    
    def setLearningRate(self, learningRate):
        self.learningRate = learningRate


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
            logging.debug("")
            logging.debug("--------- Feedforward Layer " + str(i) + "/" + str(self.nLayers-1) + str("-------------"))
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

    def computeError(self, networkOutput, target):
        # SquaredErrors will be the array with the errors 
        # for each output neuron against the desired
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
                partialDeltas[i, j] = -(target[j] - neuron)*layer.activationDerivative(neuron)
                # print("-(target[j] - neuron)", -(target[j] - neuron))
                # print("neuron*(1 - neuron)", neuron*(1 - neuron))
                # print("neuronPrev", neuronPrev)
                # print("neuron", neuron)
                # print("target", target[j])
                # print(neuronPrev*partialDeltas[i, j], -(target[j] - neuron)*neuronPrev*neuron*(1 - neuron))
                # w' = w - eta*∂C/∂w | ∂C/∂w = ∂z/∂w*(∂a/∂z*∂C/∂a), (∂a/∂z*∂C/∂a) = partialDelta
                weights[i, j] = weights[i, j] - self.learningRate*(neuronPrev*partialDeltas[i, j])
        #         print("new weight:", weights[i, j])
        logging.debug("New weights output layer:")
        logging.debug(weights)
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
                newPartialDeltas[i, j] = total*layer.activationDerivative(neuron)
                weights[i, j] = weights[i, j] - self.learningRate*newPartialDeltas[i, j]*neuronPrev
        logging.debug("New weights:")
        logging.debug(weights)
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
        print('Target:', target)
        print('Total cost:', totalCost)
        print('Average cost:', totalCost/len(inputData))
        return allOutputs[-1]

    
    def train(self, inputData, target, miniBatch=True):
        if (len(inputData) != len(target)):
            errorStr = 'Input and output differ in size'
            raise ValueError(errorStr)
        
        logging.debug('All targets:' + str(target))
        totalCost = 0

        # Propagate the input forward, get the outputs (also stored inside each layer)
        # print('Input', (inputData))
        # allOutputs = self.feedForward(inputData)
        # print('Output', allOutputs[-1][0])
        # print("************++")
        for j in range(0, len(inputData)):
            # print('Input', np.array(inputData[j], ndmin=2))
            allOutputs = self.feedForward(np.array(inputData[j], ndmin=2))
            # print('Output', allOutputs[-1][0])
            # for i in range(self.nLayers - 1, -1, -1):
            #     # logging.debug("")
            #     # logging.debug("--------- Backpropagation Layer" + str(i) + "/" + str(self.nLayers-1) + " -------------")
            #     layer = self.layers[i]
            #     output = layer.getOutput()[0]
            #     print(output)
            #     print("---")
            
            # exit(0)
            # Compute the squared error with the last (-1, output layer) for the input case we're treating (j)
            squaredError = self.computeError(allOutputs[-1][0], target[j])
            totalCost = totalCost + squaredError
            logging.debug("")
            logging.debug("--------- For input" + str(j) + "/" + str(len(inputData)-1) + " | Cost: " + str(squaredError) + " -------------")
            logging.debug("Output")
            logging.debug(allOutputs[-1][0])
            logging.debug("Target")
            logging.debug(target[j])
            logging.debug("Input")
            logging.debug(inputData[j])
            for i in range(self.nLayers - 1, -1, -1):
                logging.debug("")
                logging.debug("--------- Backpropagation Layer" + str(i) + "/" + str(self.nLayers-1) + " -------------")
                layer = self.layers[i]
                output = layer.getOutput()[0]
                # print(layer.getOutput())
                # exit(0)

                if i == 0:
                    outputPrev = inputData[j]
                else:
                    layerPrev = self.layers[i-1]
                    outputPrev = layerPrev.getOutput()[0]

                if i == self.nLayers - 1:
                    partialDeltas = self.computeWeightsOutputLayer(outputPrev, output, target[j], layer)
                else:
                    partialDeltas = self.computeWeights(outputPrev, output, layer, self.layers[i+1], partialDeltas)
            
            # Apply the new weights for this input case
            logging.debug('Applying new weights...')
            for layer in self.layers:
                layer.applyNewWeights()
            
            # print('Cost for last input:', squaredError)

        logging.debug('Total cost:' + str(totalCost))
        return totalCost


        
