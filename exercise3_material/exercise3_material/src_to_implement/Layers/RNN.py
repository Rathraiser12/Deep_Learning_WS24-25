import numpy as np
import copy
from .Base import BaseLayer
from .Sigmoid import Sigmoid
from .TanH import TanH
from .FullyConnected import FullyConnected


# python -m src_to_implement.NeuralNetworkTests Bonus

class RNN(BaseLayer):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.trainable = True

        # Whether to carry hidden state from one forward call to the next
        self.memorize_flag = False

        # Optimizer reference
        self.optimizer_val = None

        # Basic dimensions
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Hidden states over time
        self.hidden_states = [np.zeros((1, self.hidden_dim))]

        # Internal layers
        self.hidden_fc = FullyConnected(in_dim + hidden_dim, hidden_dim)
        self.tanh_act = TanH()
        self.sigmoid_act = Sigmoid()
        self.output_fc = FullyConnected(hidden_dim, out_dim)

        # Placeholders for gradients
        self.grad_hidden_fc = None
        self.grad_output_fc = None

        # to keep references to weights for external access if needed
        self.weights = self.hidden_fc.weights
        self.weights_hy = self.output_fc.weights

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fc.initialize(weights_initializer, bias_initializer)
        self.output_fc.initialize(weights_initializer, bias_initializer)
        self.weights = self.hidden_fc.weights
        self.weights_hy = self.output_fc.weights

    def forward(self, input_data):
        # Debug: Starting RNN forward pass here
        if not self.memorize:
            self.hidden_states = [np.zeros((1, self.hidden_dim))]

        self.input_data = input_data
        self.batch_count = input_data.shape[0]

        # Container for final outputs (predictions)
        self.predictions = np.zeros((self.batch_count, self.out_dim))

        # Unroll through time
        for t, single_input in enumerate(input_data):
            current_x = single_input.reshape(1, -1)
            prev_h = self.hidden_states[-1].reshape(1, -1)

            # Combine input and previous hidden state
            concat_input = np.hstack([current_x, prev_h])

            # Compute new hidden state
            raw_hidden = self.hidden_fc.forward(concat_input)
            new_h = self.tanh_act.forward(raw_hidden)
            self.hidden_states.append(new_h)

            # Compute output
            raw_output = self.output_fc.forward(new_h)
            self.predictions[t] = self.sigmoid_act.forward(raw_output)

        # Debug: Forward pass completed here
        return self.predictions

    def backward(self, error_signal):
        #back propogation starts here
        # Debug: Starting RNN backward pass
        self.grad_hidden_fc = np.zeros_like(self.hidden_fc.weights)
        self.grad_output_fc = np.zeros_like(self.output_fc.weights)

        # Gradients wrt RNN input for each timestep
        input_grads = np.zeros((self.batch_count, self.in_dim))

        # Gradient wrt hidden state (initialized to zero)
        grad_h = np.zeros((1, self.hidden_dim))

        # Process timesteps in reverse order
        for t in reversed(range(error_signal.shape[0])):
            single_err = error_signal[t, :]

            # Re-run forward pass to restore correct activations
            # (necessary for hwne not caching intermediate values)
            curr_x = self.input_data[t, :].reshape(1, -1)
            prev_h = self.hidden_states[t].reshape(1, -1)
            concat_input = np.hstack([curr_x, prev_h])
            self.sigmoid_act.forward(
                self.output_fc.forward(
                    self.tanh_act.forward(
                        self.hidden_fc.forward(concat_input)
                    )
                )
            )

            # Backprop through sigmoid and output FC
            local_grad = self.sigmoid_act.backward(single_err)
            local_grad = self.output_fc.backward(local_grad) + grad_h
            self.grad_output_fc += self.output_fc.gradient_weights

            # Backprop through tanh and hidden FC
            local_grad = self.tanh_act.backward(local_grad)
            local_grad = self.hidden_fc.backward(local_grad)
            self.grad_hidden_fc += self.hidden_fc.gradient_weights

            # Split gradient into the portion for input and the portion for hidden state
            input_grads[t, :] = local_grad[:, :self.in_dim]
            grad_h = local_grad[:, self.in_dim:]

        # Update weights using optimizer if available
        if self.optimizer is not None:
            self.hidden_fc.weights = self.optimizer.calculate_update(
                self.hidden_fc.weights, self.grad_hidden_fc
            )
            self.output_fc.weights = self.optimizer.calculate_update(
                self.output_fc.weights, self.grad_output_fc
            )

        # Refresh references to updated weights
        self.weights = self.hidden_fc.weights
        self.weights_hy = self.output_fc.weights

        # Debug: Backward pass completed
        return input_grads

    @property
    def memorize(self):
        return self.memorize_flag

    @memorize.setter
    def memorize(self, value):
        self.memorize_flag = value

    @property
    def weights(self):
        return self.hidden_fc.weights

    @weights.setter
    def weights(self, w):
        self.hidden_fc.weights = w

    @property
    def gradient_weights(self):
        return self.grad_hidden_fc

    @gradient_weights.setter
    def gradient_weights(self, new_grad):
        self.hidden_fc._gradient_weights = new_grad

    @property
    def optimizer(self):
        return self.optimizer_val

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self.optimizer_val = copy.deepcopy(new_optimizer)
