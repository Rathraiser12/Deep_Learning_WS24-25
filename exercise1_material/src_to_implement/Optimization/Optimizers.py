import numpy as np

class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = np.atleast_1d(weight_tensor)
        gradient_tensor = np.atleast_1d(gradient_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor
