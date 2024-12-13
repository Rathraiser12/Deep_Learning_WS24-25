import numpy as np

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        if isinstance(stride_shape, int):
            stride_shape = (stride_shape,)
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.input_tensor = None
        self._optimizer = None
        self.bias_optimizer = None

        if len(convolution_shape) == 3:
            in_channels, kernel_height, kernel_width = convolution_shape
            self.weights = np.random.randn(num_kernels, in_channels, kernel_height, kernel_width) * 0.1
        elif len(convolution_shape) == 2:
            in_channels, kernel_size = convolution_shape
            self.weights = np.random.randn(num_kernels, in_channels, kernel_size) * 0.1
        else:
            raise ValueError("Convolution shape must be either 1D or 2D.")

        self.bias = np.random.randn(num_kernels) * 0.1

        self.gradient_weights = None
        self.gradient_bias = None
        self.pad_y_left = self.pad_y_right = self.pad_x_left = self.pad_x_right = 0
        self.pad_left = self.pad_right = 0

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]

        if len(self.convolution_shape) == 3:
            in_channels, kernel_height, kernel_width = self.convolution_shape
            if len(self.stride_shape) == 2:
                stride_y, stride_x = self.stride_shape
            else:
                stride_y = stride_x = self.stride_shape[0]

            H_in, W_in = input_tensor.shape[2], input_tensor.shape[3]

            self.pad_y_left = (kernel_height - 1) // 2
            self.pad_y_right = (kernel_height - 1) - self.pad_y_left
            self.pad_x_left = (kernel_width - 1) // 2
            self.pad_x_right = (kernel_width - 1) - self.pad_x_left

            input_padded = np.pad(
                input_tensor,
                ((0, 0), (0, 0),
                 (self.pad_y_left, self.pad_y_right),
                 (self.pad_x_left, self.pad_x_right)),
                mode='constant', constant_values=0
            )

            out_height = (H_in - kernel_height + self.pad_y_left + self.pad_y_right) // stride_y + 1
            out_width = (W_in - kernel_width + self.pad_x_left + self.pad_x_right) // stride_x + 1

            output = np.zeros((batch_size, self.num_kernels, out_height, out_width))

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for i in range(out_height):
                        for j in range(out_width):
                            h_start = i * stride_y
                            w_start = j * stride_x
                            region = input_padded[b, :, h_start:h_start + kernel_height, w_start:w_start + kernel_width]
                            output[b, k, i, j] = np.sum(region * self.weights[k]) + self.bias[k]

            return output

        else:
            in_channels, kernel_size = self.convolution_shape
            stride = self.stride_shape[0]
            W_in = input_tensor.shape[2]
            self.pad_left = (kernel_size - 1) // 2
            self.pad_right = (kernel_size - 1) - self.pad_left

            input_padded = np.pad(
                input_tensor,
                ((0, 0), (0, 0),
                 (self.pad_left, self.pad_right)),
                mode='constant', constant_values=0
            )

            out_width = (W_in - kernel_size + self.pad_left + self.pad_right) // stride + 1
            output = np.zeros((batch_size, self.num_kernels, out_width))

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for i in range(out_width):
                        w_start = i * stride
                        region = input_padded[b, :, w_start:w_start + kernel_size]
                        output[b, k, i] = np.sum(region * self.weights[k]) + self.bias[k]

            return output

    def backward(self, error_tensor):
        if len(self.convolution_shape) == 3:
            in_channels, kernel_height, kernel_width = self.convolution_shape
            batch_size = self.input_tensor.shape[0]
            _, _, H_in, W_in = self.input_tensor.shape
            if len(self.stride_shape) == 2:
                stride_y, stride_x = self.stride_shape
            else:
                stride_y = stride_x = self.stride_shape[0]
            input_padded = np.pad(
                self.input_tensor,
                ((0, 0), (0, 0),
                 (self.pad_y_left, self.pad_y_right),
                 (self.pad_x_left, self.pad_x_right)),
                mode='constant', constant_values=0
            )

            _, _, out_height, out_width = error_tensor.shape

            self.gradient_weights = np.zeros_like(self.weights)
            self.gradient_bias = np.zeros_like(self.bias)
            grad_input_padded = np.zeros_like(input_padded)

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for i in range(out_height):
                        for j in range(out_width):
                            err_val = error_tensor[b, k, i, j]
                            h_start = i * stride_y
                            w_start = j * stride_x
                            region = input_padded[b, :, h_start:h_start + kernel_height, w_start:w_start + kernel_width]
                            self.gradient_weights[k] += region * err_val
                            self.gradient_bias[k] += err_val
                            grad_input_padded[b, :, h_start:h_start + kernel_height, w_start:w_start + kernel_width] += self.weights[k] * err_val
            grad_input = grad_input_padded[:, :, self.pad_y_left:self.pad_y_left + H_in, self.pad_x_left:self.pad_x_left + W_in]

        else:
            in_channels, kernel_size = self.convolution_shape
            batch_size = self.input_tensor.shape[0]
            _, _, W_in = self.input_tensor.shape

            stride = self.stride_shape[0]
            input_padded = np.pad(
                self.input_tensor,
                ((0, 0), (0, 0),
                 (self.pad_left, self.pad_right)),
                mode='constant', constant_values=0
            )
            _, _, out_width = error_tensor.shape

            self.gradient_weights = np.zeros_like(self.weights)
            self.gradient_bias = np.zeros_like(self.bias)
            grad_input_padded = np.zeros_like(input_padded)

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for i in range(out_width):
                        err_val = error_tensor[b, k, i]
                        w_start = i * stride
                        region = input_padded[b, :, w_start:w_start + kernel_size]
                        self.gradient_weights[k] += region * err_val
                        self.gradient_bias[k] += err_val
                        grad_input_padded[b, :, w_start:w_start + kernel_size] += self.weights[k] * err_val
            grad_input = grad_input_padded[:, :, self.pad_left:self.pad_left + W_in]


        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return grad_input

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels

        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels,), fan_in=1, fan_out=1)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        if optimizer is not None:
            optimizer_params = vars(optimizer).copy()
            try:
                self.bias_optimizer = optimizer.__class__(**optimizer_params)
            except TypeError:
                self.bias_optimizer = optimizer.__class__(
                    optimizer.learning_rate,
                    getattr(optimizer, 'mu', None),
                    getattr(optimizer, 'rho', None)
                )
        else:
            self.bias_optimizer = None