# main.py

import numpy as np
from Layers.FullyConnected import FullyConnected
from Optimization.Optimizers import Sgd

# Initialize the optimizer
optimizer = Sgd(learning_rate=0.01)

# Create an instance of the FullyConnected layer
fc_layer = FullyConnected(input_size=4, output_size=3)
fc_layer.optimizer = optimizer

# Create a dummy input tensor
input_tensor = np.random.rand(5, 4)  # Batch size of 5, input size of 4

# Perform a forward pass
output = fc_layer.forward(input_tensor)
print("Forward Output:", output)

# Create a dummy error tensor for backward pass
error_tensor = np.random.rand(5, 3)  # Batch size of 5, output size of 3

# Perform a backward pass
error_to_previous = fc_layer.backward(error_tensor)
print("Backward Output:", error_to_previous)
