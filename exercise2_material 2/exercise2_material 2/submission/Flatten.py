class Flatten:
    def __init__(self):
        self.input_shape = None
        self.trainable = False  # DEBUG: Flatten layer does not have weights, hence not trainable

    def forward(self, input_tensor):
        # Storing the input shape so we can "unflatten" during backpropagation
        self.input_shape = input_tensor.shape
        #Reshaping to (batch_size, number_of_features)
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):
        # DEBUG: Reshaping the error tensor back to the original input shape to ensures that error dimensions match the upstream layer.
        return error_tensor.reshape(self.input_shape)
        # print("Backward Flatten - reshaped error tensor:", error_tensor.shape)
        # return error_tensor.reshape(self.input_shape)
