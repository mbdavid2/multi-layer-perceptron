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

def setLoggingLevel(args):
    logLevel = 'None'
    if len(args) >= 2:
        logLevel = args[1]

    if logLevel == '--debug':
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

def plotError(errorByLearningRate):
    print(errorByLearningRate)

    for rate, error in errorByLearningRate.items():
        plt.plot(error, '*', label='Mean squared error with η =' + str(rate))
        # plt.set(xlabel="Iteration", ylabel="Mean squared error average")
        plt.xlabel('Iteration')
        plt.ylabel('Mean squared error')
    plt.legend(loc="upper left")

    # ax4.set_ylim(0, 180)
    
    # plt.suptitle(paramConfigString, fontsize=14)
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0]*4, Size[1]*4, forward=True)  # Set forward to True to resize window along with plot in figure.
    plt.show()
    # plt.savefig(resultsFolder + 'plot' + paramConfigurationKey + '.png')

def getLetter(vector):
    index = np.argmax(vector)
    return chr(index + A_CODE)

def getOutputVector(letter):
    # A's code is 65, start indexing at that
    letterVector = np.zeros(26, dtype=np.int8)
    index = ord(letter) - A_CODE
    letterVector[index] = 1
    return letterVector
    # print(letterVector)
    # print(letter, ord(letter), chr(ord(letter)))


def loadLetterDataset(filename, testSize, training, trainingTarget, test, testTarget):
    random.seed(2)
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        for line in lines:
            # Decide if it is going to be for the training or the testing set
            if random.random() > testSize:
                features = []
                # Get the letter label and the features
                
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


def multipleLearningRates(network, inputData, outputData, iterations, letters, rates = [0.15, 0.25, 0.5, 0.75, 1, 1.5, 1.75, 2]):
    trainData, testData = inputData
    trainTarget, testTarget = outputData
    errorByLearningRate = {}
    for learningRate in rates:
        iterationError = []
        network.reset()
        network.setLearningRate(learningRate)
        print("Learning rate:", learningRate)
        for i in range(0, iterations):
            logging.info("--------------- Iteration " + str(i) + " | Input size: " + str(len(trainData)) + " ---------------")
            error = network.train(trainData, trainTarget)
            (allOutputs, totalError) = network.test(testData, testTarget, printOutput=False)

            ######
            if letters:
                nCorrect = 0
                allOutputs = network.test(testData, testTarget, printOutput=False)
                for i, output in enumerate(allOutputs):
                    target = getLetter(trainTarget[i])
                    output = getLetter(output)
                    # print("Target:", target, "| Output:", output) 
                    if target == output:
                        nCorrect = nCorrect + 1

                print("Accuracy:", nCorrect/len(allOutputs))
                error = nCorrect/len(allOutputs)
        
            ######

            if np.isnan(error):
                iterationError = []
                break
            else:
                iterationError.append(totalError)

        if len(iterationError) != 0:
            errorByLearningRate[learningRate] = iterationError
        
        network.test(testData, testTarget, printOutput=False)
    
    plotError(errorByLearningRate)

# Using http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
def trainLetterRecognition():
    training = []
    trainingTarget = []
    test = []
    testTarget = []
    loadLetterDataset('letter-recognition.data', 0.30, training, trainingTarget, test, testTarget)
    training = np.array(training)
    trainingTarget = np.array(trainingTarget)
    test = np.array(test)
    testTarget = np.array(testTarget)

    # for i in range(0, len(test)-1):
    #     print(test[i])
    #     print(testTarget[i])
    #     print("--------------")
    # print(test)
    # print(testTarget)
    # exit()

    # Setting up the network, 16 input units for the 16 given features
    hiddenLayer = Layer.Layer(16, 50)
    hiddenLayer2 = Layer.Layer(50, 30)
    # 26 outputs for each of the alphabet letters
    outputLayer = Layer.Layer(30, 26)

    network = Network.Network([hiddenLayer, hiddenLayer2, outputLayer], learningRate=0.25)
    # multipleLearningRates(network, training, trainingTarget, 5, True, rates = [0.01, 0.1, 0.25, 0.5, 1, 2, 3])
    # exit()
    
    globalStart = time.time()
    for i in range(0, 4):
        start = time.time()
        print("--------------- Iteration", i, "input size:", len(training), "---------------")
        network.train(training, trainingTarget)
        nCorrect = 0
        (allOutputs, totalError) = network.test(test, testTarget, printOutput=False)
        for i, output in enumerate(allOutputs):
            target = getLetter(testTarget[i])
            output = getLetter(output)
            # print("Target:", target, "| Output:", output) 
            if target == output:
                nCorrect = nCorrect + 1

        print("Accuracy:", nCorrect/len(allOutputs))
        end = time.time()
        print("Finished iteration with time:", (end - start)/60, "minutes")

    # allOutputs = network.test(test, testTarget)




    # nCorrect = 0
    # for i, output in enumerate(allOutputs):
    #     target = getLetter(testTarget[i])
    #     output = getLetter(output)
    #     print("Target:", target, "| Output:", output) 
    #     if target == output:
    #         nCorrect = nCorrect + 1

    # print("Accuracy:", nCorrect/len(allOutputs))
    globalEnd = time.time()
    print("Finished training with time:", (globalEnd - globalStart)/60, "minutes") 

def trainXOR():
    hiddenLayer = Layer.Layer(2, 7)
    outputLayer = Layer.Layer(7, 1)

    inputData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    desiredOutput = np.array([[0], [1], [1], [0]])

    network = Network.Network([hiddenLayer, outputLayer], learningRate=0.95)
    for i in range(0, 2000):
        error = network.train(inputData, desiredOutput)
        (allOutputs, totalError) = network.test(inputData, desiredOutput, printOutput=True)

    exit()
    multipleLearningRates(network, inputData, desiredOutput, 1000) #, rates=[1.5])

def trainSinus():
    trainData = []
    target = []
    testData = []
    testTarget = []
    random.seed(2)
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
    outputLayer = Layer.Layer(7, 1, activation=Neuron.Same())

    network = Network.Network([hiddenLayer, outputLayer])
    multipleLearningRates(network, (trainData, testData), (target, testTarget), 200, False, rates = [0.001, 0.01, 0.1, 0.2, 0.3,0.5, 0.75,1,2,3])

def main(args):
    setLoggingLevel(args)
    # trainSinus()
    # exit()
    # trainXOR()
    trainLetterRecognition()


if __name__ == '__main__':
    sys.exit(main(sys.argv))




