import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size, weights_initializer=None, bias_initializer=None):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        # Initialize weights including bias
        if self.weights_initializer is not None and self.bias_initializer is not None:
            self.initialize_weights()
        else:
            # Initialize weights with uniform distribution between 0 and 1, including bias as the last row
            self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))

        # Private attributes for optimizer and gradients
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
        """
        Perform the forward pass of the fully connected layer.
        """
        batch_size = input_tensor.shape[0]

        # Append a column of ones to the input tensor to account for the bias
        ones = np.ones((batch_size, 1))
        extended_input = np.hstack((input_tensor, ones))

        self.input_tensor = extended_input  # Store for backward pass

        # Compute the output
        output = np.dot(extended_input, self.weights)

        return output

    def backward(self, error_tensor):
        """
        Perform the backward pass of the fully connected layer.
        """
        # Compute gradients with respect to weights (including bias)
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        # Update weights if optimizer is set
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            # Note: Bias is part of the weights (last row), so it's updated automatically

        # Compute error tensor for the previous layer (excluding bias weights)
        error_to_previous = np.dot(error_tensor, self.weights[:-1, :].T)

        return error_to_previous

    def initialize_weights(self):
        """
        Initialize weights and bias using the provided initializers.
        """
        if self.weights_initializer is None or self.bias_initializer is None:
            raise ValueError("Initializers must be provided for weights and bias.")

        # Initialize weights (excluding bias) using positional arguments
        # Assuming weights_initializer expects (shape, fan_in, fan_out)
        weights = self.weights_initializer.initialize(
            (self.input_size, self.output_size),
            self.input_size,
            self.output_size
        )

        # Initialize bias using positional arguments
        # Assuming bias_initializer expects (shape, fan_in, fan_out)
        bias = self.bias_initializer.initialize(
            (1, self.output_size),
            self.input_size,
            self.output_size
        )

        # Combine weights and bias into a single weight matrix
        self.weights = np.vstack([weights, bias])

    def initialize(self, weights_initializer, bias_initializer):
        """
        Public method to set initializers and initialize weights.
        """
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.initialize_weights()
