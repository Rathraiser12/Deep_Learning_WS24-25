import numpy as np

class L1_Regularizer:
    def __init__(self, alpha):
        # Debug: Initializing L1_Regularizer
        self.rate = alpha  # internally store alpha as rate

    def calculate_gradient(self, weights):
        # Debug: Calculating L1 gradient
        # print("L1 gradient calculation")
        return self.rate * np.sign(weights)

    def norm(self, weights):
        # Debug: Calculating L1 norm
        return self.rate * np.sum(np.abs(weights))


class L2_Regularizer:
    def __init__(self, alpha):
        # Debug: Initializing L2_Regularizer
        self.rate = alpha

    def calculate_gradient(self, weights):
        # Debug: Calculating L2 gradient
        return self.rate * weights

    def norm(self, weights):
        # Debug: Calculating L2 norm
        return self.rate * np.sum(np.square(weights))
