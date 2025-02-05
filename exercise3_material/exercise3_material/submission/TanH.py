from .Base import BaseLayer
import numpy as np

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        # Debug: TanH layer initialized
        self.output_cache = None

    def forward(self, in_data):
        # Debug: TanH forward pass
        self.output_cache = np.tanh(in_data)
        # print("TanH forward output:", self.output_cache)
        return self.output_cache

    def backward(self, error_signal):
        # Debug: TanH backward pass
        return error_signal * (1.0 - np.square(self.output_cache))
