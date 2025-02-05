import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # Store prediction tensor for backward pass
        self.prediction_tensor = prediction_tensor
        # Using machine epsilon for numerical stability as the test suite expects this epsilon instead of any other ones
        epsilon = np.finfo(float).eps

        prediction_clipped = np.clip(prediction_tensor, epsilon, 1. - epsilon)
        # Debug: Clipped predictions to avoid log(0)

        loss = -np.sum(label_tensor * np.log(prediction_clipped))

        return loss

    def backward(self, label_tensor):
        epsilon = np.finfo(float).eps
        prediction_clipped = np.clip(self.prediction_tensor, epsilon, 1. - epsilon)

        # Compute gradient
        error_tensor = - (label_tensor / prediction_clipped)
        # print(f"Error tensor (gradient): {error_tensor}")

        return error_tensor
