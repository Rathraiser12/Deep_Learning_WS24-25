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

        # Add bias here, check the file once
        if self.weights_initializer is not None and self.bias_initializer is not None:
            self.initialize_weights()
        else:
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
        """
        Perform the forward pass of the fully connected layer.
        """
        batch_size = input_tensor.shape[0]
        ones = np.ones((batch_size, 1))
        extended_input = np.hstack((input_tensor, ones))
        self.input_tensor = extended_input 
        output = np.dot(extended_input, self.weights)

        return output

    def backward(self, error_tensor):
        """
        Perform the backward pass of the fully connected layer.
        """
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        error_to_previous = np.dot(error_tensor, self.weights[:-1, :].T)

        return error_to_previous

    def initialize_weights(self):
        """
        Initialize weights and bias using the provided initializers.
        """
        if self.weights_initializer is None or self.bias_initializer is None:
            raise ValueError("Initializers must be provided for weights and bias.")
        weights = self.weights_initializer.initialize(
            (self.input_size, self.output_size),
            self.input_size,
            self.output_size
        )
        bias = self.bias_initializer.initialize(
            (1, self.output_size),
            self.input_size,
            self.output_size
        )
        self.weights = np.vstack([weights, bias])

    def initialize(self, weights_initializer, bias_initializer):
        """
        Public method to set initializers and initialize weights.
        """
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.initialize_weights()