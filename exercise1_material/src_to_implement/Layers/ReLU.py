# Layers/ReLU.py

import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        # ReLU has no trainable parameters
        self.input_tensor = None  # To store input for backward pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor  # Store input for backward pass
        output_tensor = np.maximum(0, input_tensor)
        return output_tensor

    def backward(self, error_tensor):
        # Compute derivative of ReLU
        derivative = np.where(self.input_tensor > 0, 1, 0)
        # Multiply element-wise with the incoming error tensor
        error_to_previous = error_tensor * derivative
        return error_to_previous
