import numpy as np
import mainClasses as mlp
import Layer
import Network
import random
import csv

# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]

# np.random.seed(0)
# weights = np.random.rand(4, 4)

# biases = [2, 3, 0.5]

# np.random.seed(0)
# biases = np.random.rand(1, 4)

# inputs = [[1, 2, 3],
#           [2.0, 5.0, -1.0, 2.0],
#           [-1.5, 2.7, 3.3, -0.8]]

# output = np.dot(inputs, np.array(weights).T) + biases
# print(output)


# sigmoid = mlp.ActivationFunction()
# nNeurons = 4
# neurons = []
# for i in range(0, nNeurons):
#     neurons.append(mlp.Neuron(sigmoid))

# print(neurons)

# for neuron in neurons:
#     print(neuron.activate(3))
A_CODE = 65

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
    hiddenLayer = Layer.Layer(16, 30, True)
    hiddenLayer2 = Layer.Layer(30, 30, True)
    # 26 outputs for each of the alphabet letters
    outputLayer = Layer.Layer(30, 26, False)
    
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


        

trainLetterRecognition()

# hiddenLayer = Layer.Layer(2, 4, True)
# # hiddenLayer2 = mlp.Layer(3, 3)
# outputLayer = Layer.Layer(4, 1, False)
# # outputLayer = mlp.Layer(hiddenLayer.getLayerSize(), 3)

# inputData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# desiredOutput = np.array([[0], [1], [1], [0]])

# # inputData = np.array([[0.05, 0.10]])
# # desiredOutput = np.array([[0.01, 0.99]])

# network = Network.Network([hiddenLayer, outputLayer])
# # network.feedForward(inputData)
# for i in range(0, 2000):
#     print("--------------- Iteration", i, "---------------")
#     network.train(inputData, desiredOutput)

# network.test(inputData, desiredOutput)
