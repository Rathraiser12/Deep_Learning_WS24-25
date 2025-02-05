import numpy as np
from .Base import BaseLayer
from .Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):
    def __init__(self, channels: int):
        super().__init__()
        self.trainable = True
        self.testing_phase = False
        self.channels = channels
        # Gamma and beta parameters (weights and bias)
        self.weights = np.ones(channels, dtype=float)
        self.bias = np.zeros(channels, dtype=float)

        # Running statistics
        self.running_mean = np.zeros(channels, dtype=float)
        self.running_var = np.ones(channels, dtype=float)
        self.momentum = 0.0  # Per test requirements

        self.epsilon = 1e-12  # Avoid division by zero

        # Cache for forward and backward pass
        self.input_tensor = None
        self.batch_mean = None
        self.batch_var = None
        self.normalized_input = None
        self.gradient_weights = None
        self.gradient_bias = None

        # Store the original shape for reformat
        self.original_shape = None
     #print("[DEBUG] BatchNormalization layer initialized with channels =", channels)

    def initialize(self, weights_initializer, bias_initializer):
      # Using number of channels as a dummy value for fan_in and fan_out
        fan_in = self.channels
        fan_out = self.channels

        # Initialize weights and bias
        self.weights = weights_initializer.initialize((self.channels,), fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.channels,), fan_in, fan_out)
      # print("[DEBUG] initialize() called in BatchNormalization")

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor
        x_2d, orig_shape = self._flatten_if_needed(input_tensor)
        self.original_shape = orig_shape  # Store the original shape for reformatting
        # print("[DEBUG] forward() called with input shape:", input_tensor.shape)
        if not self.testing_phase:
          # print("[DEBUG] Training phase BN forward pass")
            self.batch_mean = np.mean(x_2d, axis=0)
            self.batch_var = np.var(x_2d, axis=0)
            self.normalized_input = (x_2d - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            out_2d = self.weights * self.normalized_input + self.bias

            # Update running stats
            self.running_mean = self.batch_mean
            self.running_var = self.batch_var
        else:
          # print("[DEBUG] Testing phase BN forward pass using running statistics")
            x_hat = (x_2d - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out_2d = self.weights * x_hat + self.bias

        return self._reshape_back(out_2d, orig_shape)

    def reformat(self, tensor: np.ndarray) -> np.ndarray:
      # print("[DEBUG] reformat() called with tensor shape:", tensor.shape)
        if tensor.ndim == 4:
            # Flatten 4D to 2D
            b, c, h, w = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(-1, c)
        elif tensor.ndim == 2:
            if self.original_shape is None:
                raise ValueError("Cannot infer reshaping for 2D tensors without original context.")
            # Reshape 2D back to 4D using stored original shape
            return self._reshape_back(tensor, self.original_shape)
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
      # print("[DEBUG] backward() called with error_tensor shape:", error_tensor.shape)
        err_2d, orig_shape = self._flatten_if_needed(error_tensor)
        x_2d, _ = self._flatten_if_needed(self.input_tensor)

        dgamma = np.sum(err_2d * self.normalized_input, axis=0)
        dbeta = np.sum(err_2d, axis=0)

        self.gradient_weights = dgamma
        self.gradient_bias = dbeta
      # print("[DEBUG] Gradient w.r.t. gamma:", dgamma)
      # print("[DEBUG] Gradient w.r.t. beta:", dbeta)
        if hasattr(self, 'optimizer') and self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, dgamma)
            self.bias = self.optimizer.calculate_update(self.bias, dbeta)

        dx_2d = compute_bn_gradients(
            error_tensor=err_2d,
            input_tensor=x_2d,
            weights=self.weights,
            mean=self.batch_mean,
            var=self.batch_var,
            eps=self.epsilon
        )

        return self._reshape_back(dx_2d, orig_shape)

    def _flatten_if_needed(self, tensor: np.ndarray):
        shape = tensor.shape
        if tensor.ndim == 4:
            b, c, h, w = shape
            flattened = tensor.transpose(0, 2, 3, 1).reshape(-1, c)
            # print(f"[DEBUG] _flatten_if_needed(): 4D -> 2D, from {shape} to {flattened.shape}")
            return flattened, shape
        elif tensor.ndim == 2:
          # print(f"[DEBUG] _flatten_if_needed(): Already 2D with shape {shape}")
            return tensor, shape
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    def _reshape_back(self, flattened: np.ndarray, original_shape) -> np.ndarray:
        if len(original_shape) == 2:
          # print(f"[DEBUG] _reshape_back(): Returning 2D tensor with shape {flattened.shape}")
            return flattened
        elif len(original_shape) == 4:
            b, c, h, w = original_shape
            return flattened.reshape(b, h, w, c).transpose(0, 3, 1, 2)
        # print(f"[DEBUG] _reshape_back(): Reshaping to 4D tensor with shape {reshaped.shape}")
        else:
            raise ValueError(f"Cannot reshape back to original shape: {original_shape}")
