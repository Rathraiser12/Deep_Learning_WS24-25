import numpy as np
from .Base import BaseLayer

class Dropout(BaseLayer):
    #Uses inverted dropout to preserve expected activation values.
    def __init__(self, keep_rate: float):
        super().__init__()
        if not 0 < keep_rate < 1:
            raise ValueError("keep_rate must be between 0 and 1.")
        self.keep_rate = keep_rate
        self.drop_mask = None  # Will store the dropout mask during training

    def forward(self, in_data: np.ndarray) -> np.ndarray:
        # Debug: Entering Dropout forward pass
        if not self.testing_phase:
            # Training mode
            self.drop_mask = (np.random.rand(*in_data.shape) < self.keep_rate).astype(np.float32)
            dropped_out = in_data * self.drop_mask / self.keep_rate
            return dropped_out
        else:
            # Evaluation mode
            return in_data

    def backward(self, out_grad: np.ndarray) -> np.ndarray:
        # Debug: Entering Dropout backward pass
        if not self.testing_phase:
            # Training mode
            grad_input = out_grad * self.drop_mask / self.keep_rate
            return grad_input
        else:
            # Evaluation mode
            return out_grad
