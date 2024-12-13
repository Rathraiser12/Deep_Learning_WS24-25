# Layers/SoftMax.py

import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        # Subtract max for numerical stability
        shifted_logits = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted_logits)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        self.output_tensor = exp_shifted / sum_exp  # Store for backward pass
        # print(f"Output tensor (SoftMax probabilities): {self.output_tensor}")

        return self.output_tensor

    def backward(self, error_tensor):
        # print(f"Error tensor shape: {error_tensor.shape}")

        # Compute dot product between error_tensor and output_tensor, sum over classes
        dot_product = np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True)

        error_to_previous = self.output_tensor * (error_tensor - dot_product)

        return error_to_previous
