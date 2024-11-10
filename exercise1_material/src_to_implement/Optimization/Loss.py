# Optimization/Loss.py

import numpy as np
'''The implementation of CrossEntropyLoss is sus. especially the way it manipulates the epsilon value 
to specifically pass the test case. Look into it and understand how it works and make the changes accordingly'''
class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None  # To store predictions from forward pass

    def forward(self, prediction_tensor, label_tensor):
        # Store prediction tensor for backward pass
        self.prediction_tensor = prediction_tensor

        # Use machine epsilon for numerical stability
        epsilon = np.finfo(float).eps
        prediction_clipped = np.clip(prediction_tensor, epsilon, 1. - epsilon)

        # Compute cross-entropy loss accumulated over the batch
        loss = -np.sum(label_tensor * np.log(prediction_clipped))
        return loss

    def backward(self, label_tensor):
        # Use machine epsilon for numerical stability
        epsilon = np.finfo(float).eps
        prediction_clipped = np.clip(self.prediction_tensor, epsilon, 1. - epsilon)

        # Compute gradient
        error_tensor = - (label_tensor / prediction_clipped)
        return error_tensor
