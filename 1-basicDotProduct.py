import numpy as np

inputs = [1, 2, 3, 2.5]
# 3 weight sets for each of the 3 output neurons
weights1 = [0.2, 0.8, -0.5, 1.0] # weights from the 4 input neurons to the output neuron 1
weights2 = [0.5, -0.91, 0.26, -0.5] # weights from the 4 input neurons to the output neuron 2
weights3 = [-0.26, -0.27, 0.17, 0.87] # weights from the 4 input neurons to the output neuron 3

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = []


# 3 outputs, each is the input values from the 4 previous neurons, by the weights
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + biases[0],
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + biases[1],
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + biases[2]
         ]
print(output)

print(np.dot(weights, inputs) + biases)
