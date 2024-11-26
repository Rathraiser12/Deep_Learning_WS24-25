import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        # Initializing weights including bias with uniform distribution between 0 and 1
        self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))
        # print(f"Initialized weights with shape {self.weights.shape}: {self.weights}")

        self._optimizer = None
        self._gradient_weights = None


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def forward(self, input_tensor):
        # print("Forward pass started.")
        # print(f"Input tensor shape: {input_tensor.shape}")

        batch_size = input_tensor.shape[0]
        ones = np.ones((batch_size, 1))
        extended_input = np.hstack((input_tensor, ones))

        self.input_tensor = extended_input  # Store for backward pass

        output = np.dot(extended_input, self.weights)
        # print(f"Output shape: {output.shape}")
        # print(f"Output: {output}")

        return output

    def backward(self, error_tensor):
        # print("Backward pass started.")
        # print(f"Error tensor shape: {error_tensor.shape}")

        # Computing gradients with respect to weights
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # print(f"Gradient weights shape: {self._gradient_weights.shape}")

        # Update weights if optimizer is set
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            # Debug: Updated weights
        else:
            # if no optimizer is set then pass
            pass

        # Computing error tensor for previous layer exccluding baias weights
        error_to_previous = np.dot(error_tensor, self.weights[:-1, :].T)
        # print(f"Error to previous layer shape: {error_to_previous.shape}")

        return error_to_previous
