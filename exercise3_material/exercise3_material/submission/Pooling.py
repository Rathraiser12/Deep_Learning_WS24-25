import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        if isinstance(stride_shape, int):
            stride_shape = (stride_shape,)
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False  

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, height, width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        if len(self.stride_shape) == 2:
            stride_height, stride_width = self.stride_shape
        else:
            stride_height = stride_width = self.stride_shape[0]
        out_height = (height - pool_height) // stride_height + 1
        out_width = (width - pool_width) // stride_width + 1
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)

        # print(f"Input tensor shape: {input_tensor.shape}") 
        # print(f"Output tensor shape: {output.shape}")     

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * stride_height
                        h_end = h_start + pool_height
                        w_start = j * stride_width
                        w_end = w_start + pool_width
                        region = input_tensor[b, c, h_start:h_end, w_start:w_end]

                        # print(f"Pooling region: {region}")              
                        max_val = np.max(region)
                        output[b, c, i, j] = max_val
                        # print(f"Max value: {max_val} at region ({b},{c},{i},{j})") 

                        max_position = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[b, c, i, j] = (h_start + max_position[0], w_start + max_position[1])
                        # print(f"Max position: {self.max_indices[b, c, i, j]}") 

        return output

    def backward(self, error_tensor):
        grad_input = np.zeros_like(self.input_tensor)
        batch_size, channels, out_height, out_width = error_tensor.shape

        # print(f"Error tensor shape: {error_tensor.shape}") # Debug
        # print(f"Gradient input shape: {grad_input.shape}") # Debug

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_max, w_max = self.max_indices[b, c, i, j]
                        grad_input[b, c, h_max, w_max] += error_tensor[b, c, i, j]
                        # print(f"Gradient assigned at position ({h_max},{w_max})") # Debug

        return grad_input