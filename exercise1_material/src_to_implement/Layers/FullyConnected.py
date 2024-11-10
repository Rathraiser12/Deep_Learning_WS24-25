import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights including bias
        self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))

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
        batch_size = input_tensor.shape[0]
        ones = np.ones((batch_size, 1))
        extended_input = np.hstack((input_tensor, ones))

        self.input_tensor = extended_input  # Store for backward pass

        output = np.dot(extended_input, self.weights)
        return output

    def backward(self, error_tensor):
        # Compute gradients
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        # Update weights if optimizer is set
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        # Compute error tensor for previous layer (exclude bias weights)
        error_to_previous = np.dot(error_tensor, self.weights[:-1, :].T)
        return error_to_previous
