import numpy as np
import mainClasses as mlp

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

np.random.seed(0)
weights = np.random.rand(4, 4)

biases = [2, 3, 0.5]

np.random.seed(0)
biases = np.random.rand(1, 4)

inputs = [[1, 2, 3],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

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

hiddenLayer = mlp.Layer(3, 4)
# outputLayer = mlp.Layer(hiddenLayer.getLayerSize(), 3)

network = mlp.Network([hiddenLayer])
network.feedForward(inputs[0])

# layer.forwardPropagation(inputs)
# print(layer.getOutput())
