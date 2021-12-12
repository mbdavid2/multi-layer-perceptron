import numpy as np
import Layer
import Network
import Neuron
import random
import csv
import matplotlib.pyplot as plt
import logging
import sys
import time

A_CODE = 65
RANDOM_SEED = 2

def setLoggingLevel(args):
    logLevel = 'None'
    if len(args) >= 2:
        logLevel = args[1]
    logUserSet = False
    if logLevel == '--debug':
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
        logUserSet = True
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logUserSet

def getLetter(vector):
    index = np.argmax(vector)
    return chr(index + A_CODE)

def getOutputVector(letter):
    # A's code is 65, start indexing at that
    letterVector = np.zeros(26, dtype=np.int8)
    index = ord(letter) - A_CODE
    letterVector[index] = 1
    return letterVector

def loadLetterDataset(filename, testSize, training, trainingTarget, test, testTarget):
    random.seed(RANDOM_SEED)
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        for line in lines:
            # Decide if it is going to be for the training or the testing set
            if random.random() > testSize:
                features = []
                # Get the letter label and then the features
                trainingTarget.append(getOutputVector(line[0]))
                for i in range(1, 17):
                    features.append(int(line[i]))
                training.append(features)
            else:
                features = []
                testTarget.append(getOutputVector(line[0]))
                for i in range(1, 17):
                    features.append(int(line[i]))
                test.append(features)

def plotError(errorByLearningRate, title, titleSimple, problemName, letters=False):
    # print(errorByLearningRate)

    for rate, error in errorByLearningRate.items():
        if letters:
            plt.ylabel('Accuracy')
            label = 'Accuracy with η =' + str(rate)
        else:
            plt.ylabel('Averaged Mean squared error')
            label = 'Mean squared error with η =' + str(rate)
        plt.plot(error, '-', label=label)
        # plt.set(xlabel="Iteration", ylabel="Mean squared error average")
        plt.xlabel('Iteration')
        
    plt.legend(loc="upper left")
    
    plt.suptitle(title, fontsize=14)
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0]*2, Size[1]*2, forward=True)  # Set forward to True to resize window along with plot in figure.
    plt.savefig('results/' + problemName + '_' + titleSimple + '.png')
    plt.show()

def plotAccDiffNetworks(networks, errorByNetwork, problemName):
    print(errorByNetwork)

    for network in networks:
        simpleCode = network.getLayersDescriptionSimple()
        errorList = errorByNetwork[simpleCode]
        plt.ylabel('Accuracy')
        label = 'Accuracy with ' + str(simpleCode)
        plt.plot(errorList, '-', label=label)
        # plt.set(xlabel="Iteration", ylabel="Mean squared error average")
        plt.xlabel('Iteration')        
    plt.legend(loc="upper left")
    
    # plt.suptitle(title, fontsize=14)
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0]*2, Size[1]*2, forward=True)  # Set forward to True to resize window along with plot in figure.
    plt.savefig('results/final0.2/' + problemName + '_' + "multiple" + '.png')
    # plt.show()

def getLetterAccuracy(allOutputs, testTarget, printOutput=False):
    nCorrect = 0
    for i, output in enumerate(allOutputs):
        target = getLetter(testTarget[i])
        output = getLetter(output)
        if printOutput:
            print("Target:", target, "| Output:", output) 
        if target == output:
            nCorrect = nCorrect + 1
    accuracy = nCorrect/len(allOutputs)
    # print("Accuracy:", accuracy)
    return accuracy

def multipleNetworks(networks, inputData, outputData, iterations, letters, problemName):
    (trainData, testData) = inputData
    (trainTarget, testTarget) = outputData
    learningRate = 0.2
    errorByNetwork = {}
    accuracyByNetwork = {}
    for network in networks:
        iterationError = []
        iterationAccuracy = []
        network.reset()
        network.setLearningRate(learningRate)
        print("Learning rate:", learningRate)
        for i in range(0, iterations):
            logging.info("--------------- Iteration " + str(i) + " | Input size: " + str(len(trainData)) + " ---------------")
            if letters and i < 5:
                network.setLearningRate(0.5)
            else:
                network.setLearningRate(learningRate)
            error = network.train(trainData, trainTarget)
            (allOutputs, totalError) = network.test(testData, testTarget, printOutput=False)

            # If the problem is letters, print accuracy
            if letters:
                accuracy = getLetterAccuracy(allOutputs, testTarget)
                iterationAccuracy.append(accuracy)

            if np.isnan(error):
                iterationError = []
                iterationAccuracy = []
                break
            else:
                iterationError.append(totalError)

        if len(iterationError) != 0:
            errorByNetwork[network.getLayersDescriptionSimple()] = iterationError
            accuracyByNetwork[network.getLayersDescriptionSimple()] = iterationAccuracy
    
    # Print accuracy
    plotAccDiffNetworks(networks, accuracyByNetwork, problemName)


def multipleLearningRates(network, inputData, outputData, iterations, letters, problemName, rates = [0.1, 0.25, 0.5, 0.75, 1, 2]):
    (trainData, testData) = inputData
    (trainTarget, testTarget) = outputData
    
    errorByLearningRate = {}
    accuracyByLearningRate = {}
    for learningRate in rates:
        iterationError = []
        iterationAccuracy = []
        network.reset()
        network.setLearningRate(learningRate)
        print("Learning rate:", learningRate)
        for i in range(0, iterations):
            logging.info("--------------- Iteration " + str(i) + " | Input size: " + str(len(trainData)) + " ---------------")
            if letters and i < 10:
                network.setLearningRate(0.5)
            else:
                network.setLearningRate(learningRate)
            error = network.train(trainData, trainTarget)
            (allOutputs, totalError) = network.test(testData, testTarget, printOutput=False)

            # If the problem is letters, print accuracy
            if letters:
                accuracy = getLetterAccuracy(allOutputs, testTarget)
                iterationAccuracy.append(accuracy)

            if np.isnan(error):
                iterationError = []
                iterationAccuracy = []
                break
            else:
                iterationError.append(totalError)

        if len(iterationError) != 0:
            errorByLearningRate[learningRate] = iterationError
            accuracyByLearningRate[learningRate] = iterationAccuracy
    
    # Print loss
    plotError(errorByLearningRate, network.getLayersDescription(), 
              network.getLayersDescriptionSimple(), problemName, False)

    # If letters, print accuracy too              
    if letters:
        problemName = problemName + "accu"
        plotError(accuracyByLearningRate, network.getLayersDescription(), 
                network.getLayersDescriptionSimple(), problemName, letters)


def generateDifferentNetworks():
    hiddenLayerA = Layer.Layer(16, 36, smaller=True)
    hiddenLayerA2 = Layer.Layer(36, 36, smaller=True)
    outputLayerA = Layer.Layer(36, 26, smaller=True)
    networkA = Network.Network([hiddenLayerA, hiddenLayerA2, outputLayerA], learningRate=0.07)

    hiddenLayerB = Layer.Layer(16, 36, smaller=True)
    outputLayerB = Layer.Layer(36, 26, smaller=True)
    networkB = Network.Network([hiddenLayerB, outputLayerB], learningRate=0.07)

    hiddenLayerC = Layer.Layer(16, 36, smaller=True)
    hiddenLayerC2 = Layer.Layer(36, 50, smaller=True)
    outputLayerC = Layer.Layer(50, 26, smaller=True)
    networkC = Network.Network([hiddenLayerC, hiddenLayerC2, outputLayerC], learningRate=0.07)

    hiddenLayerD = Layer.Layer(16, 70, smaller=True)
    hiddenLayerD2 = Layer.Layer(70, 60, smaller=True)
    outputLayerD = Layer.Layer(60, 26, smaller=True)
    networkD = Network.Network([hiddenLayerD, hiddenLayerD2, outputLayerD], learningRate=0.07)

    hiddenLayerE = Layer.Layer(16, 80, smaller=True)
    outputLayerE = Layer.Layer(80, 26, smaller=True)
    networkE = Network.Network([hiddenLayerE, outputLayerE], learningRate=0.07)
    
    return [networkA, networkB, networkC, networkD, networkE]

# Using http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
def trainLetterRecognition(plot=False):
    training = []
    trainingTarget = []
    test = []
    testTarget = []
    loadLetterDataset('datasets/letter-recognition.data', 1/5, training, trainingTarget, test, testTarget)
    training = np.array(training)
    trainingTarget = np.array(trainingTarget)
    test = np.array(test)
    testTarget = np.array(testTarget)

    # Setting up the network, 16 input units for the 16 given features
    hiddenLayer = Layer.Layer(16, 70, smaller=True)
    hiddenLayer2 = Layer.Layer(70, 100, smaller=True)
    # 26 outputs for each of the alphabet letters
    outputLayer = Layer.Layer(100, 26, smaller=True)

    network = Network.Network([hiddenLayer,hiddenLayer2, outputLayer], learningRate=0.07)
    network.printNetworkInfo()
    inputData = (training, test)
    outputData = (trainingTarget, testTarget)
    if plot:
        networks = generateDifferentNetworks()
        # multipleLearningRates(network, inputData, outputData, 1000, True, "letters", rates=[0.01, 0.1, 0.2, 0.5, 0.75, 1]) #rates=[0.1, 0.05, 0.01]
        multipleNetworks(networks, inputData, outputData, 1000, True, "letters")
    else:
        rates = [0.15]
        globalStart = time.time()
        errorByLearningRate = {}
        iterationError = []
        for i in range(0, 1000):
            start = time.time()
            print("--------------- Iteration", i, "input size:", len(training), "---------------")
            if i < len(rates):
                learningRate = rates[i]
            else:
                learningRate = rates[-1]
            network.setLearningRate(learningRate)
            if i < 5:
                network.setLearningRate(0.5)
            if i > 100:
                network.setLearningRate(0.1)
            network.train(training, trainingTarget) 

            (allOutputsTest, testError) = network.test(test, testTarget, printOutput=False)
            accuracyTest = getLetterAccuracy(allOutputsTest, testTarget, False)
            (allOutputs, totalErrorTest) = network.test(training, trainingTarget, printOutput=False)
            accuracyTrain = getLetterAccuracy(allOutputs, trainingTarget, False)
            print("Test accuracy:", accuracyTest, "| Train accuracy:", accuracyTrain)
            print("Test loss:", testError, "| Train loss:", totalErrorTest)
            totalError = accuracyTest
        
            if np.isnan(totalError):
                iterationError = []
                break
            else:
                iterationError.append(totalError)

            end = time.time()
            # print("Finished iteration with time:", (end - start)/60, "minutes")

        accuracy = getLetterAccuracy(allOutputsTest, testTarget, False)
        if len(iterationError) != 0:
            errorByLearningRate[learningRate] = iterationError
        plotError(errorByLearningRate, network.getLayersDescription(), 
            network.getLayersDescriptionSimple(), "letters", True)

        globalEnd = time.time()
        print("Finished training with time:", (globalEnd - globalStart)/60, "minutes")
        print("Test accuracy:", accuracyTest, "| Train accuracy:", accuracyTrain)
        print("Test loss:", testError, "| Train loss:", totalErrorTest)

        # (allOutputs, totalError) = network.test(test, testTarget, printOutput=False)
        # print("train data:")
        # (allOutputs, totalError) = network.test(training, trainingTarget, printOutput=False)

def trainXOR(plot=False):
    hiddenLayer = Layer.Layer(2, 4)
    outputLayer = Layer.Layer(4, 1)
    network = Network.Network([hiddenLayer, outputLayer], learningRate=1)
    network.printNetworkInfo()

    inputData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    desiredOutput = np.array([[0], [1], [1], [0]])
    
    # In this case use same input/output for training and testing
    allInput = (inputData, inputData)
    allOutput = (desiredOutput, desiredOutput)

    if not plot:
        for i in range(0, 2000):
            print('--------- Iteration', i, '----------')
            error = network.train(inputData, desiredOutput)
            (allOutputs, totalError) = network.test(inputData, desiredOutput, printOutput=True)
            print("-------------------")
    else:
        multipleLearningRates(network, allInput, allOutput, 5000, False, "xor", rates=[0.01, 0.25, 0.5, 0.75, 1])

def trainSinus(plot=False):
    trainData = []
    target = []
    testData = []
    testTarget = []

    # Create dataset (400 for training and 100 for testing)
    random.seed(RANDOM_SEED)
    for j in range(0, 500):
        v = []
        for i in range(0, 4):
            u = random.uniform(-1, 1)
            v.append(u)
        combination = np.sin(v[0]-v[1]+v[2]-v[3])
        if j < 400:
            trainData.append(v)
            target.append([combination])
        else:
            testData.append(v)
            testTarget.append([combination])
    
    trainData = np.array(trainData)
    target = np.array(target)
    testData = np.array(testData)
    testTarget = np.array(testTarget)

    hiddenLayer = Layer.Layer(4, 7)
    outputLayer = Layer.Layer(7, 1, activation=Neuron.Identity())

    network = Network.Network([hiddenLayer, outputLayer], learningRate = 0.75)
    if not plot:
        for i in range(0, 2000):
            print('--------- Iteration', i, '----------')
            error = network.train(trainData, target)
            (allOutputs, totalError) = network.test(testData, testTarget, printOutput=False)
        (allOutputs, totalError) = network.test(testData, testTarget, printOutput=False)
        print("train data:")
        (allOutputs, totalError) = network.test(trainData, target, printOutput=False)

    else:
        multipleLearningRates(network, (trainData, testData), (target, testTarget), 100, False, "sin", rates = [0.001, 0.01, 0.1, 0.2, 0.3,0.5,0.75,1])

def printUsage():
    print("Usage: python3 main.py [--debug] [xor/sin/letters (plot)] ")
    exit()

def main(args):
    # Decide which problem to execute and debugging level. Simple "parser"
    logUserSet = setLoggingLevel(args)
    if (logUserSet and len(args) > 2) or (not logUserSet and len(args) >= 2):
        if logUserSet:
            problem = args[2]
        else:
            problem = args[1]
        if len(args) >= 3:
            plot = args[2]
            if plot == 'plot':
                plot = True
            else:
                printUsage()
        else:
            plot = False

        if problem == 'xor':
            trainXOR(plot)
        elif problem == 'sin':
            trainSinus(plot)
        elif problem == 'letters':
            trainLetterRecognition(plot)
        else:
            printUsage()
    else:
        trainLetterRecognition()

if __name__ == '__main__':
    sys.exit(main(sys.argv))




