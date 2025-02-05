from .Base import BaseLayer
import numpy as np

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        # Debug: Sigmoid layer initialized
        self.output_cache = None

    def forward(self, in_data):
        # Debug: Sigmoid forward pass
        self.output_cache = 1.0 / (1.0 + np.exp(-in_data))
        # print("Forward Sigmoid output:", self.output_cache)
        return self.output_cache

    def backward(self, error_signal):
        # Debug: Sigmoid backward pass
        return self.output_cache * (1.0 - self.output_cache) * error_signal
