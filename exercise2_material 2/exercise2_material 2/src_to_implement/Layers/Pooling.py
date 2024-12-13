import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        # Ensure stride_shape is a tuple
        if isinstance(stride_shape, int):
            stride_shape = (stride_shape,)
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False  # Pooling layers are not trainable

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, height, width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape

        # Determine stride for height and width
        if len(self.stride_shape) == 2:
            stride_height, stride_width = self.stride_shape
        else:
            stride_height = stride_width = self.stride_shape[0]

        # Calculate output dimensions
        out_height = (height - pool_height) // stride_height + 1
        out_width = (width - pool_width) // stride_width + 1

        # Initialize output tensor
        output = np.zeros((batch_size, channels, out_height, out_width))

        # Initialize max indices to store the positions of maxima
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * stride_height
                        h_end = h_start + pool_height
                        w_start = j * stride_width
                        w_end = w_start + pool_width

                        # Extract the current pooling region
                        region = input_tensor[b, c, h_start:h_end, w_start:w_end]

                        # Find the maximum value and its indices within the region
                        max_val = np.max(region)
                        output[b, c, i, j] = max_val

                        # Get the relative indices of the maximum value
                        # In case of multiple maxima, np.argmax returns the first occurrence
                        max_position = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[b, c, i, j] = (h_start + max_position[0], w_start + max_position[1])

        return output

    def backward(self, error_tensor):
        grad_input = np.zeros_like(self.input_tensor)
        batch_size, channels, out_height, out_width = error_tensor.shape

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Get the position of the maximum value in the current pooling window
                        h_max, w_max = self.max_indices[b, c, i, j]

                        # Assign the gradient to the position of the maximum value
                        grad_input[b, c, h_max, w_max] += error_tensor[b, c, i, j]

        return grad_input
