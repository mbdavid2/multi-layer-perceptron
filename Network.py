import numpy as np
import logging

from numpy.core.numeric import full

class Network:
    def __init__(self, layers, softMax=False, learningRate = 0.5):
        self.layers = layers
        self.nLayers = len(layers)
        self.learningRate = learningRate
        self.useSoftMax = softMax


    def checkDimensions(self, inputData, layer, layerN):
        # Check that the input size is the same as layers[0]
        if inputData.shape[1] != layer.getInputSize():
            errorStr = ('Stopped at Layer(' + str(layerN) + ') -> Incoming input size: ' 
                        + str(inputData.shape[1]) + ' != ' + str(layer.getInputSize()) + 
                       ' (Layer(' + str(layerN) + ') expected input size)')
            raise ValueError(errorStr)

    
    def getLayersDescription(self):
        description = "Network:"
        for i, layer in enumerate(self.layers):
            if i == 0:
                description = description + " (Input " + str(layer.inputSize) + ")->"
                description = description + layer.getActivationName() + "(" + str(layer.size)
            else:
                description = description + layer.getActivationName() + "(" + str(layer.size)

            if i < len(self.layers) - 1:
                description = description + ")->"
            else:
                description = description + ", SoftMax: " + str(self.useSoftMax) + ")"
        return description

    def getLayersDescriptionSimple(self):
        description = ""
        for i, layer in enumerate(self.layers):
            if i == 0:
                description = description + "in" + str(layer.inputSize) + "_"
                description = description + layer.getActivationName(True) + str(layer.size)
            else:
                description = description + layer.getActivationName(True) + str(layer.size)

            if i < len(self.layers) - 1:
                description = description + "_"
            else:
                description = description + "soft_" + str(self.useSoftMax).lower()[0]

        return description


    def printNetworkInfo(self):
        for i, layer in enumerate(self.layers):
            print("Layer", i, " | Input size:", layer.inputSize, "| Neurons:", layer.size)


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
        totalSums = np.sum(exponents, axis=1, keepdims=True)
        normValues = exponents / totalSums
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
            if self.useSoftMax and i == len(self.layers) - 1:
                nextInput = self.softMax(nextInput)
            allOutputs.append(nextInput)

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
        weights = layer.getWeightsRef()
        biases = layer.getBiasesRef()

        partialDeltas = np.array([(output - target)*layer.activationDerivative(output)])
        out = np.array([outputPrev])
        fullGradient = np.dot(out.T, partialDeltas)
        newWeights = weights - self.learningRate*fullGradient
        newBiases = biases - self.learningRate*partialDeltas
        layer.prepareNewParameters(newWeights, newBiases)
        return partialDeltas

    # ∂C/∂a = (a - target)
    #       = 1/2(target - out)^2, power and 1/2 cancel each other out
    # ∂a/∂z = a*(1 - a) (derivative of sigmoid)
    #       f(x) = 1/(1+e^-x), then f'(x) = f(x)(1-f(x))
    # ∂z/∂w = a_prev (previous activated output)
    #       = w*a_prev (this one) + w*a_prev (other weights) + b = a_prev + 0 + 0
    def computeWeightsOutputLayerIterative(self, outputPrev, output, target, layer):
        # C = total error (C_0 + C_1 + ... C_n), a = activated output
        # z = raw output, w = weight
        # ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a
        weights = layer.getWeightsCopy()
        biases = layer.getBiasesCopy()
        partialDeltas = np.zeros(weights.shape)
        for i, neuronPrev in enumerate(outputPrev):
            for j, neuron in enumerate(output):
                # We need: ∂z/∂w*∂a/∂z*∂C/∂a. Store ∂a/∂z*∂C/∂a for later
                partialDeltas[i, j] = -(target[j] - neuron)*layer.activationDerivative(neuron)
               
                # w' = w - eta*∂C/∂w | ∂C/∂w = ∂z/∂w*(∂a/∂z*∂C/∂a), (∂a/∂z*∂C/∂a) = partialDelta
                weights[i, j] = weights[i, j] - self.learningRate*(neuronPrev*partialDeltas[i, j])

                if i == 0:
                    biases[i, j] = biases[i, j] - self.learningRate*(partialDeltas[i, j])

        logging.debug("New weights output layer:")
        logging.debug(weights)
        layer.prepareNewParameters(weights, biases)
        return partialDeltas

    def computeWeights(self, outputPrev, output, layer, nextLayer, partialDeltas):
        weights = layer.getWeightsCopy()
        biases = layer.getBiasesCopy()
        nextWeightsRef = nextLayer.weights

        newPartialDeltas =  np.dot(partialDeltas, nextWeightsRef.T)*layer.activationDerivative(output)
        out = np.array([outputPrev])
        fullGradient = np.dot(out.T, newPartialDeltas)
        newWeights = weights - self.learningRate*fullGradient
        newBiases = biases - self.learningRate*newPartialDeltas
        layer.prepareNewParameters(newWeights, newBiases)
        return newPartialDeltas

    def computeWeightsIterative(self, outputPrev, output, layer, nextLayer, partialDeltas):
        weights = layer.getWeightsCopy()
        biases = layer.getBiasesCopy()
        nextWeightsRef = nextLayer.weights

        newPartialDeltas = np.zeros(weights.shape)
        for i, neuronPrev in enumerate(outputPrev):
            for j, neuron in enumerate(output):
                # ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a
                # ∂C/∂a = "total" = ∂C/∂a = ∂C_0/∂a + ... + ∂C_n/∂a
                total = 0
                for t in range(0, partialDeltas.shape[1]):
                    # The neuron affects multiple neurons on the next layer
                    # ∂C_i/∂a = ∂C_i/∂z*∂z/∂a = a*b, index with t instead of i for the output
                    a = partialDeltas[0, t] # a = ∂C_i/∂z = ∂C_i/∂a*∂a/∂z, already computed
                    b = nextWeightsRef[j, t] # b = ∂z/∂a = w]
                    total = total + a*b
                # We need: ∂C/∂w = ∂z/∂w*∂a/∂z*∂C/∂a = ∂z/∂w*∂a/∂z*total. Store ∂a/∂z*total for later
                newPartialDeltas[i, j] = total*layer.activationDerivative(neuron)
                weights[i, j] = weights[i, j] - self.learningRate*newPartialDeltas[i, j]*neuronPrev
                if i == 0:
                    biases[i, j] = biases[i, j] - self.learningRate*newPartialDeltas[i, j]
        logging.debug("New weights:")
        logging.debug(weights)
        layer.prepareNewParameters(weights, biases)
        return newPartialDeltas

    def test(self, inputData, target, printOutput=True):   
        print('Testing with input size:', len(inputData))
        print('Running feedforward...')
        allOutputs = self.feedForward(inputData)

        # Compute new error
        totalCost = 0
        for j in range(0, len(inputData)):
            squaredError = self.computeError(allOutputs[-1][j], target[j])
            totalCost = totalCost + squaredError
            if printOutput:
                print('Output ' + str(j) + ' ' + str(allOutputs[-1][j]) + ' | Target: ' + str(target[j]))
        
        averagedCost = totalCost/len(inputData)
        print('Total cost:', totalCost)
        print('Averaged cost:', averagedCost)
        return (allOutputs[-1], averagedCost)

    
    def train(self, inputData, target):
        logging.info("Training with learning rate: " + str(self.learningRate))
        if (len(inputData) != len(target)):
            errorStr = 'Input and output differ in size'
            raise ValueError(errorStr)
        
        logging.debug('All targets:' + str(target))
        totalCost = 0

    
        for j in range(0, len(inputData)):
            allOutputs = self.feedForward(np.array(inputData[j], ndmin=2))
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
                layer.applyNewParameters()
            
            # print('Cost for last input:', squaredError)

        averagedCost = totalCost/len(inputData)
        logging.info('Total cost during training: ' + str(totalCost))
        logging.info('Average cost: ' + str(averagedCost))
        return averagedCost


        
