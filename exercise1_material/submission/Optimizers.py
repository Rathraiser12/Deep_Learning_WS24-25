import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # ensuring weight_tensor is at least 1-dimensional
        weight_tensor = np.atleast_1d(weight_tensor)
        # ensuring gradient_tensor is at least 1-dimensional
        gradient_tensor = np.atleast_1d(gradient_tensor)
        # print(f"Weight tensor shape: {weight_tensor.shape}")
        # print(f"Gradient tensor shape: {gradient_tensor.shape}")

        updated_weights = weight_tensor - self.learning_rate * gradient_tensor

        return updated_weights

