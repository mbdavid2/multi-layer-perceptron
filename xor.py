import numpy as np
import mainClasses as mlp

sigmoid = mlp.ActivationFunction()
nNeurons = 4
neurons = []
for i in range(0, nNeurons):
    neurons.append(mlp.Neuron(sigmoid))

print(neurons)

for neuron in neurons:
    print(neuron.activate(3))

layer = mlp.Layer(4, neurons)
layer.forwardPropagation(inputs)
print(layer.getOutput())
