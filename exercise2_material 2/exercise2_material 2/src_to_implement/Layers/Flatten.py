class Flatten:
    def __init__(self):
        self.input_shape = None
        self.trainable = False  # Ensure this matches expectations

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)