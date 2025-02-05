import copy
import numpy as np
from scipy.signal import convolve, correlate
from .Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        # Debug: Initializing Conv layer
        self.is_trainable = True
        self.stride_dims = stride_shape
        self.kernel_dims = convolution_shape
        self.kernel_count = num_kernels
        self.is_2d_stride = (len(self.stride_dims) == 2)

        # Initialize weights and bias
        self.weights = np.random.rand(self.kernel_count, *self.kernel_dims)
        self.bias = np.random.rand(self.kernel_count)

        # Gradient placeholders
        self._grad_w = None
        self._grad_b = None

        # Optimizer references
        self._weight_opt = None
        self._bias_opt = None

    def forward(self, fwd_in_data):
        # Debug: Conv forward pass
        self.input_tensor = fwd_in_data
        self.batch_size, self.in_channels, *spatial_dims = self.input_tensor.shape

        # Prepare output array
        self.fwd_out_data = np.zeros((self.batch_size, self.kernel_count, *spatial_dims))

        # Compute correlation for each image in batch, for each kernel
        for b_idx, single_image in enumerate(self.input_tensor):
            for k_idx, single_kernel in enumerate(self.weights):
                self.fwd_out_data[b_idx, k_idx] = correlate(
                    single_image, single_kernel, mode='same'
                )[self.in_channels // 2]
                self.fwd_out_data[b_idx, k_idx] += self.bias[k_idx]

        # Downsample based on stride
        if self.is_2d_stride:
            return self.fwd_out_data[
                :, :, ::self.stride_dims[0], ::self.stride_dims[1]
            ]
        else:
            return self.fwd_out_data[:, :, ::self.stride_dims[0]]

    def backward(self, bwd_in_error):
        # Debug: Conv backward pass
        upsampled_grad = np.zeros_like(self.fwd_out_data)

        # Place error values in correct positions based on stride
        if self.is_2d_stride:
            upsampled_grad[:, :, ::self.stride_dims[0], ::self.stride_dims[1]] = bwd_in_error
        else:
            upsampled_grad[:, :, ::self.stride_dims[0]] = bwd_in_error

        # Prepare kernel for gradient computation
        temp_grad_kernel = np.swapaxes(self.weights, 1, 0)
        temp_grad_kernel = np.flip(temp_grad_kernel, axis=1)

        # Initialize the gradient wrt the input
        bwd_out_error = np.zeros_like(self.input_tensor)

        # Compute input gradients via convolution
        for elem_idx, grad_elem in enumerate(upsampled_grad):
            for k_idx, flipped_k in enumerate(temp_grad_kernel):
                bwd_out_error[elem_idx, k_idx] = convolve(
                    grad_elem, flipped_k, mode="same"
                )[self.kernel_count // 2]

        # If 2D, pad input before calculating gradients wrt weights
        if self.is_2d_stride:
            left_pad = (self.kernel_dims[1] - 1) // 2
            right_pad = self.kernel_dims[1] // 2
            top_pad = (self.kernel_dims[2] - 1) // 2
            bottom_pad = self.kernel_dims[2] // 2
            self.input_tensor = np.pad(
                self.input_tensor, ((0, 0), (0, 0), (left_pad, right_pad), (top_pad, bottom_pad))
            )
        else:
            left_pad = (self.kernel_dims[1] - 1) // 2
            right_pad = self.kernel_dims[1] // 2
            self.input_tensor = np.pad(
                self.input_tensor, ((0, 0), (0, 0), (left_pad, right_pad))
            )

        # Reset gradients for weights and biases
        self._grad_w = np.zeros_like(self.weights)
        self._grad_b = np.zeros_like(self.bias)

        # Compute gradients wrt weights and biases
        for elem_idx, elem_err in enumerate(upsampled_grad):
            for chan_idx, chan_err in enumerate(elem_err):
                for inp_chan_idx in range(self.kernel_dims[0]):
                    self._grad_w[chan_idx, inp_chan_idx] += correlate(
                        self.input_tensor[elem_idx, inp_chan_idx], chan_err, mode='valid'
                    )
                self._grad_b[chan_idx] += np.sum(chan_err)

        # Update weights and biases if optimizers exist
        if self._weight_opt is not None:
            self.weights = self._weight_opt.calculate_update(self.weights, self._grad_w)
        if self._bias_opt is not None:
            self.bias = self._bias_opt.calculate_update(self.bias, self._grad_b)

        return bwd_out_error

    def initialize(self, weights_initializer, bias_initializer):
        # Debug: Initializing Conv layer weights and bias
        fan_in = np.prod(self.kernel_dims)
        fan_out = self.kernel_count * np.prod(self.kernel_dims[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def gradient_weights(self):
        return self._grad_w

    @gradient_weights.setter
    def gradient_weights(self, new_grad):
        self._grad_w = new_grad

    @property
    def gradient_bias(self):
        return self._grad_b

    @gradient_bias.setter
    def gradient_bias(self, new_bias_grad):
        self._grad_b = new_bias_grad

    @property
    def optimizer(self):
        return self._weight_opt

    @optimizer.setter
    def optimizer(self, new_optimizer):
        # Debug: Setting optimizer for weights and bias
        self._weight_opt = copy.deepcopy(new_optimizer)
        self._bias_opt = copy.deepcopy(new_optimizer)
