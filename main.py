import numpy as np
import mainClasses as mlp

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

hiddenLayer = mlp.Layer(2, 2, True)
# hiddenLayer2 = mlp.Layer(3, 3)
outputLayer = mlp.Layer(2, 2, False)
# outputLayer = mlp.Layer(hiddenLayer.getLayerSize(), 3)

inputData = np.array([[0.05, 0.10]])
desiredOutput = np.array([[0.01, 0.99]])
network = mlp.Network([hiddenLayer, outputLayer])
# network.feedForward(inputData)
for i in range(0, 100):
    network.train(inputData, desiredOutput)
