import numpy as np
import mainClasses as mlp
import Layer
import Network
import Neuron
import random
import csv
import matplotlib.pyplot as plt

A_CODE = 65

def plotError(errorByLearningRate):
    for rate, error in errorByLearningRate.items():
        plt.plot(error, '*', label='Mean squared error with η =' + str(rate))
        # plt.set(xlabel="Iteration", ylabel="Mean squared error average")
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


def multipleLearningRates(network, inputData, desiredOutput, iterations):
    errorByLearningRate = {}
    for learningRate in [0.15, 0.25, 0.5, 0.75, 1, 1.5, 1.75, 2]:
        iterationError = []
        network.reset()
        network.setLearningRate(learningRate)
        for i in range(0, iterations):
            print("--------------- Iteration", i, "---------------")
            error = network.train(inputData, desiredOutput)
            iterationError.append(error)
        errorByLearningRate[learningRate] = iterationError
        network.test(inputData, desiredOutput)
    
    plotError(errorByLearningRate)

# Using http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
def trainLetterRecognition():
    training = []
    trainingTarget = []
    test = []
    testTarget = []
    loadLetterDataset('letter-recognition.data', 2/3, training, trainingTarget, test, testTarget)
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
    hiddenLayer = Layer.Layer(16, 30)
    hiddenLayer2 = Layer.Layer(30, 30)
    # 26 outputs for each of the alphabet letters
    outputLayer = Layer.Layer(30, 26)
    
    network = Network.Network([hiddenLayer, hiddenLayer2, outputLayer])
    
    for i in range(0, 1):
        print("--------------- Iteration", i, "---------------")
        for j, example in enumerate(training):
            network.train(np.array([example]), np.array([trainingTarget[j]]))

    allOutputs = network.test(test, testTarget)

    nCorrect = 0
    for i, output in enumerate(allOutputs):
        target = getLetter(testTarget[i])
        output = getLetter(output)
        print("Target:", target, "| Output:", output) 
        if target == output:
            nCorrect = nCorrect + 1

    print("Accuracy:", nCorrect/len(allOutputs))

def trainXOR():
    hiddenLayer = Layer.Layer(2, 7)
    outputLayer = Layer.Layer(7, 1)

    inputData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    desiredOutput = np.array([[0], [1], [1], [0]])

    network = Network.Network([hiddenLayer, outputLayer])

    multipleLearningRates(network, inputData, desiredOutput, 100)


def trainSinus():
    inputData = []
    sinuses = []
    random.seed(2)
    for j in range(0, 100):
        v = []
        for i in range(0, 4):
            u = random.uniform(-1, 1)
            v.append(u)
        inputData.append(v)
        combination = np.sin(v[0]-v[1]+v[2]-v[3])
        sinuses.append([combination])
    
    inputData = np.array(inputData)
    sinuses = np.array(sinuses)
    # print(inputData)
    # print(sinuses)
    # exit()

    hiddenLayer = Layer.Layer(4, 3)
    outputLayer = Layer.Layer(3, 1, activation=Neuron.Same())

    network = Network.Network([hiddenLayer, outputLayer])
    totalCost = []
    for i in range(0, 500):
        print("--------------- Iteration", i, "---------------")
        cost = network.train(inputData, sinuses)
        totalCost.append(cost)

    network.test(inputData, sinuses)
    plotError(totalCost, )
    # print(sinuses)


# trainSinus()
# exit()
trainXOR()
# trainLetterRecognition()


