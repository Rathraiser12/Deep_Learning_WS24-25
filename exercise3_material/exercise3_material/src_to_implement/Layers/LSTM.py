import numpy as np
from .Base import BaseLayer
# from .Optimizers import Optimizer        # If needed
# from .Initializers import SomeInitializer # If needed
# etc.

class LSTM(BaseLayer):
    """
    A simple LSTM layer that:
      - Forwards over time dimension (T).
      - Caches all states (f, i, o, g, c, h).
      - BPTT in the backward pass (no partial re-forward needed).
      - Exposes a single 'weights' property that includes bias columns.
      - Zeros out gradients at the start of each backward() call.
      - Applies optimizer at the end.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trainable = True

        # -- Gate parameters --
        # We'll store them as separate matrices for readability:
        # forget gate
        self.W_xf = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size,))

        # input gate
        self.W_xi = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size,))

        # output gate
        self.W_xo = np.random.randn(hidden_size, input_size) * 0.01
        self.W_ho = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size,))

        # candidate gate
        self.W_xg = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hg = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_g = np.zeros((hidden_size,))

        # output layer from hidden to output
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size,))

        # hidden/cell states at the *end* of forward pass
        self.hidden_state = np.zeros((hidden_size,))
        self.cell_state = np.zeros((hidden_size,))

        # "memorize" property: if False, resets states each forward
        self._memorize = False

        # forward caches
        self.x_s = []
        self.f_s = []
        self.i_s = []
        self.o_s = []
        self.g_s = []
        self.c_s = []
        self.h_s = []
        self.y_s = []

        # gradient accumulators
        self.grad_W_xf = np.zeros_like(self.W_xf)
        self.grad_W_hf = np.zeros_like(self.W_hf)
        self.grad_b_f  = np.zeros_like(self.b_f)

        self.grad_W_xi = np.zeros_like(self.W_xi)
        self.grad_W_hi = np.zeros_like(self.W_hi)
        self.grad_b_i  = np.zeros_like(self.b_i)

        self.grad_W_xo = np.zeros_like(self.W_xo)
        self.grad_W_ho = np.zeros_like(self.W_ho)
        self.grad_b_o  = np.zeros_like(self.b_o)

        self.grad_W_xg = np.zeros_like(self.W_xg)
        self.grad_W_hg = np.zeros_like(self.W_hg)
        self.grad_b_g  = np.zeros_like(self.b_g)

        self.grad_W_hy = np.zeros_like(self.W_hy)
        self.grad_b_y  = np.zeros_like(self.b_y)

        self.optimizer = None  # If you want to apply parameter updates

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, val):
        self._memorize = val
        if not val:
            self.hidden_state = np.zeros((self.hidden_size,))
            self.cell_state   = np.zeros((self.hidden_size,))

    def forward(self, input_tensor):
        """
        Args:
          input_tensor: shape (T, input_size)
        Returns:
          output_sequence: shape (T, output_size)
        """
        T, in_size = input_tensor.shape
        if in_size != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {in_size}")

        # If not memorizing, reset hidden/cell at start
        if not self.memorize:
            self.hidden_state = np.zeros((self.hidden_size,))
            self.cell_state   = np.zeros((self.hidden_size,))

        # Clear caches
        self.x_s.clear()
        self.f_s.clear()
        self.i_s.clear()
        self.o_s.clear()
        self.g_s.clear()
        self.c_s.clear()
        self.h_s.clear()
        self.y_s.clear()

        output_sequence = np.zeros((T, self.output_size))

        for t in range(T):
            x_t = input_tensor[t]
            self.x_s.append(x_t)

            # Gate computations
            f_t = self.sigmoid(self.W_xf @ x_t + self.W_hf @ self.hidden_state + self.b_f)
            i_t = self.sigmoid(self.W_xi @ x_t + self.W_hi @ self.hidden_state + self.b_i)
            o_t = self.sigmoid(self.W_xo @ x_t + self.W_ho @ self.hidden_state + self.b_o)
            g_t = np.tanh(self.W_xg @ x_t + self.W_hg @ self.hidden_state + self.b_g)

            c_t = f_t * self.cell_state + i_t * g_t
            h_t = o_t * np.tanh(c_t)

            # Output transform
            y_t = self.W_hy @ h_t + self.b_y

            # Save states
            self.f_s.append(f_t)
            self.i_s.append(i_t)
            self.o_s.append(o_t)
            self.g_s.append(g_t)
            self.c_s.append(c_t)
            self.h_s.append(h_t)
            self.y_s.append(y_t)

            output_sequence[t] = y_t
            self.hidden_state = h_t
            self.cell_state   = c_t

        return output_sequence

    def backward(self, error_tensor):
        """
        BPTT
        error_tensor shape: (T, output_size)
        Returns:
          dX shape: (T, input_size)
        """
        T, out_size = error_tensor.shape
        if out_size != self.output_size:
            raise ValueError(f"Expected output size {self.output_size}, got {out_size}")

        # Zero-out gradient accumulators each backward
        self.grad_W_xf.fill(0)
        self.grad_W_hf.fill(0)
        self.grad_b_f.fill(0)

        self.grad_W_xi.fill(0)
        self.grad_W_hi.fill(0)
        self.grad_b_i.fill(0)

        self.grad_W_xo.fill(0)
        self.grad_W_ho.fill(0)
        self.grad_b_o.fill(0)

        self.grad_W_xg.fill(0)
        self.grad_W_hg.fill(0)
        self.grad_b_g.fill(0)

        self.grad_W_hy.fill(0)
        self.grad_b_y.fill(0)

        # Prepare the output gradient
        dX = np.zeros((T, self.input_size))

        # d_h_next, d_c_next for next iteration
        d_h_next = np.zeros((self.hidden_size,))
        d_c_next = np.zeros((self.hidden_size,))

        for t in reversed(range(T)):
            dy = error_tensor[t]  # shape (output_size,)

            # current states
            h_t = self.h_s[t]
            c_t = self.c_s[t]
            f_t = self.f_s[t]
            i_t = self.i_s[t]
            o_t = self.o_s[t]
            g_t = self.g_s[t]

            if t == 0:
                h_prev = np.zeros((self.hidden_size,))
                c_prev = np.zeros((self.hidden_size,))
            else:
                h_prev = self.h_s[t - 1]
                c_prev = self.c_s[t - 1]

            x_t = self.x_s[t]

            # Grad w.r.t. output transform
            # dW_hy, db_y, dh
            self.grad_W_hy += np.outer(dy, h_t)
            self.grad_b_y  += dy
            dh = self.W_hy.T @ dy + d_h_next

            # LSTM cell backprop
            do = dh * np.tanh(c_t)
            do_raw = do * o_t * (1 - o_t)  # derivative of sigmoid

            dc = dh * o_t * (1 - np.tanh(c_t)**2) + d_c_next

            df = dc * c_prev
            df_raw = df * f_t * (1 - f_t)

            di = dc * g_t
            di_raw = di * i_t * (1 - i_t)

            dg = dc * i_t
            dg_raw = dg * (1 - g_t**2)

            # Now accumulate gate weight grads
            # forget gate
            self.grad_W_xf += np.outer(df_raw, x_t)
            self.grad_W_hf += np.outer(df_raw, h_prev)
            self.grad_b_f  += df_raw

            # input gate
            self.grad_W_xi += np.outer(di_raw, x_t)
            self.grad_W_hi += np.outer(di_raw, h_prev)
            self.grad_b_i  += di_raw

            # output gate
            self.grad_W_xo += np.outer(do_raw, x_t)
            self.grad_W_ho += np.outer(do_raw, h_prev)
            self.grad_b_o  += do_raw

            # candidate gate
            self.grad_W_xg += np.outer(dg_raw, x_t)
            self.grad_W_hg += np.outer(dg_raw, h_prev)
            self.grad_b_g  += dg_raw

            # d h_{t-1}
            dh_prev = (self.W_hf.T @ df_raw
                       + self.W_hi.T @ di_raw
                       + self.W_ho.T @ do_raw
                       + self.W_hg.T @ dg_raw)

            # d c_{t-1}
            dc_prev = dc * f_t

            # store for next iteration
            d_h_next = dh_prev
            d_c_next = dc_prev

            # gradient wrt x_t
            dx_t = (self.W_xf.T @ df_raw
                    + self.W_xi.T @ di_raw
                    + self.W_xo.T @ do_raw
                    + self.W_xg.T @ dg_raw)
            dX[t] = dx_t

        # After BPTT, apply update if we have an optimizer
        if self.optimizer is not None:
            # forget gate
            self.W_xf = self.optimizer.calculate_update(self.W_xf, self.grad_W_xf)
            self.W_hf = self.optimizer.calculate_update(self.W_hf, self.grad_W_hf)
            self.b_f  = self.optimizer.calculate_update(self.b_f,  self.grad_b_f)

            # input gate
            self.W_xi = self.optimizer.calculate_update(self.W_xi, self.grad_W_xi)
            self.W_hi = self.optimizer.calculate_update(self.W_hi, self.grad_W_hi)
            self.b_i  = self.optimizer.calculate_update(self.b_i,  self.grad_b_i)

            # output gate
            self.W_xo = self.optimizer.calculate_update(self.W_xo, self.grad_W_xo)
            self.W_ho = self.optimizer.calculate_update(self.W_ho, self.grad_W_ho)
            self.b_o  = self.optimizer.calculate_update(self.b_o,  self.grad_b_o)

            # candidate gate
            self.W_xg = self.optimizer.calculate_update(self.W_xg, self.grad_W_xg)
            self.W_hg = self.optimizer.calculate_update(self.W_hg, self.grad_W_hg)
            self.b_g  = self.optimizer.calculate_update(self.b_g,  self.grad_b_g)

            # hidden->output
            self.W_hy = self.optimizer.calculate_update(self.W_hy, self.grad_W_hy)
            self.b_y  = self.optimizer.calculate_update(self.b_y,  self.grad_b_y)

        return dX

    # ---------------------------------------------------------------------
    # The 'weights' property
    # ---------------------------------------------------------------------
    @property
    def weights(self) -> np.ndarray:
        """
        Return a single 2D matrix of shape (input_size + hidden_size + 1, 4*hidden_size),
        storing W_x*, W_h*, b_* for the 4 gates [f, i, o, g].
        """
        # Gate f
        Wf = np.concatenate([self.W_xf, self.W_hf], axis=1)  # shape (hidden_size, input_size+hidden_size)
        Wf_b = np.concatenate([Wf, self.b_f[:, None]], axis=1)

        # Gate i
        Wi = np.concatenate([self.W_xi, self.W_hi], axis=1)
        Wi_b = np.concatenate([Wi, self.b_i[:, None]], axis=1)

        # Gate o
        Wo = np.concatenate([self.W_xo, self.W_ho], axis=1)
        Wo_b = np.concatenate([Wo, self.b_o[:, None]], axis=1)

        # Gate g
        Wg = np.concatenate([self.W_xg, self.W_hg], axis=1)
        Wg_b = np.concatenate([Wg, self.b_g[:, None]], axis=1)

        # shape => (4*hidden_size, input_size+hidden_size+1)
        W_all = np.concatenate([Wf_b, Wi_b, Wo_b, Wg_b], axis=0)
        return W_all.T  # => (input_size+hidden_size+1, 4*hidden_size)

    @weights.setter
    def weights(self, value: np.ndarray):
        exp_shape = (self.input_size + self.hidden_size + 1, 4*self.hidden_size)
        if value.shape != exp_shape:
            raise ValueError(f"Expected shape {exp_shape}, got {value.shape}")

        W_all = value.T  # shape => (4*hidden_size, input_size+hidden_size+1)
        f_block, i_block, o_block, g_block = np.split(W_all, 4, axis=0)

        def parse_block(block):
            # block has shape (hidden_size, input_size+hidden_size+1)
            Wx = block[:, :self.input_size]
            Wh = block[:, self.input_size:self.input_size+self.hidden_size]
            b_ = block[:, -1]
            return Wx, Wh, b_

        self.W_xf, self.W_hf, self.b_f = parse_block(f_block)
        self.W_xi, self.W_hi, self.b_i = parse_block(i_block)
        self.W_xo, self.W_ho, self.b_o = parse_block(o_block)
        self.W_xg, self.W_hg, self.b_g = parse_block(g_block)

    # ---------------------------------------------------------------------
    # The 'gradient_weights' property
    # ---------------------------------------------------------------------
    @property
    def gradient_weights(self) -> np.ndarray:
        """
        Must be same shape as self.weights => (input_size+hidden_size+1, 4*hidden_size)
        We'll pack our gate gradients in the same order.
        """
        # Gate f
        gf = np.concatenate([self.grad_W_xf, self.grad_W_hf], axis=1)
        gf_b = np.concatenate([gf, self.grad_b_f[:, None]], axis=1)

        # Gate i
        gi = np.concatenate([self.grad_W_xi, self.grad_W_hi], axis=1)
        gi_b = np.concatenate([gi, self.grad_b_i[:, None]], axis=1)

        # Gate o
        go = np.concatenate([self.grad_W_xo, self.grad_W_ho], axis=1)
        go_b = np.concatenate([go, self.grad_b_o[:, None]], axis=1)

        # Gate g
        gg = np.concatenate([self.grad_W_xg, self.grad_W_hg], axis=1)
        gg_b = np.concatenate([gg, self.grad_b_g[:, None]], axis=1)

        # => shape (4*hidden_size, input_size+hidden_size+1)
        G_all = np.concatenate([gf_b, gi_b, go_b, gg_b], axis=0)
        return G_all.T  # => (input_size+hidden_size+1, 4*hidden_size)

    @gradient_weights.setter
    def gradient_weights(self, dw: np.ndarray):
        exp_shape = (self.input_size + self.hidden_size + 1, 4*self.hidden_size)
        if dw.shape != exp_shape:
            raise ValueError(f"gradient_weights must have shape {exp_shape}, got {dw.shape}")
        G_all = dw.T
        f_block, i_block, o_block, g_block = np.split(G_all, 4, axis=0)

        def parse_block(block):
            gx = block[:, :self.input_size]
            gh = block[:, self.input_size:self.input_size + self.hidden_size]
            gb = block[:, -1]
            return gx, gh, gb

        self.grad_W_xf, self.grad_W_hf, self.grad_b_f = parse_block(f_block)
        self.grad_W_xi, self.grad_W_hi, self.grad_b_i = parse_block(i_block)
        self.grad_W_xo, self.grad_W_ho, self.grad_b_o = parse_block(o_block)
        self.grad_W_xg, self.grad_W_hg, self.grad_b_g = parse_block(g_block)

    # ---------------------------------------------------------------------
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def calculate_regularization_loss(self):
        """
        If your optimizer uses a .regularizer, sum up regularization for W_xf,..., W_hy etc.
        """
        reg_loss = 0.0
        if self.optimizer and hasattr(self.optimizer, 'regularizer') and self.optimizer.regularizer:
            # e.g. reg_loss += self.optimizer.regularizer.norm(self.W_xf)
            pass
        return reg_loss

    def initialize(self, weights_initializer, bias_initializer):
        """
        Re-init with given initializers.
        """
        # For example:
        # self.W_xf = weights_initializer.initialize(...)
        # self.b_f  = bias_initializer.initialize(...)
        # ...
        fan_in_input = self.input_size
        fan_in_hidden = self.hidden_size
        fan_out_hidden = self.hidden_size
        fan_out_output = self.output_size

        # forget gate
        self.W_xf = weights_initializer.initialize(
            (self.hidden_size, self.input_size),
            fan_in_input, fan_in_hidden
        )
        self.W_hf = weights_initializer.initialize(
            (self.hidden_size, self.hidden_size),
            fan_in_hidden, fan_out_hidden
        )
        self.b_f  = bias_initializer.initialize(
            (self.hidden_size,),
            fan_in_hidden, fan_out_hidden
        )

        # input gate
        self.W_xi = weights_initializer.initialize(
            (self.hidden_size, self.input_size),
            fan_in_input, fan_in_hidden
        )
        self.W_hi = weights_initializer.initialize(
            (self.hidden_size, self.hidden_size),
            fan_in_hidden, fan_out_hidden
        )
        self.b_i  = bias_initializer.initialize(
            (self.hidden_size,),
            fan_in_hidden, fan_out_hidden
        )

        # output gate
        self.W_xo = weights_initializer.initialize(
            (self.hidden_size, self.input_size),
            fan_in_input, fan_in_hidden
        )
        self.W_ho = weights_initializer.initialize(
            (self.hidden_size, self.hidden_size),
            fan_in_hidden, fan_out_hidden
        )
        self.b_o  = bias_initializer.initialize(
            (self.hidden_size,),
            fan_in_hidden, fan_out_hidden
        )

        # candidate gate
        self.W_xg = weights_initializer.initialize(
            (self.hidden_size, self.input_size),
            fan_in_input, fan_in_hidden
        )
        self.W_hg = weights_initializer.initialize(
            (self.hidden_size, self.hidden_size),
            fan_in_hidden, fan_out_hidden
        )
        self.b_g  = bias_initializer.initialize(
            (self.hidden_size,),
            fan_in_hidden, fan_out_hidden
        )

        # hidden->output
        self.W_hy = weights_initializer.initialize(
            (self.output_size, self.hidden_size),
            fan_in_hidden, fan_out_output
        )
        self.b_y  = bias_initializer.initialize(
            (self.output_size,),
            fan_in_hidden, fan_out_output
        )

    # ---------------------------------------------------------------------
    # Utility activations
    # ---------------------------------------------------------------------
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(sigmoid_out):
        return sigmoid_out * (1.0 - sigmoid_out)

    @staticmethod
    def tanh_derivative(tanh_out):
        return 1.0 - tanh_out**2
