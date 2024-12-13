import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):

        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        # print(f"[DEBUG] Forward pass input_tensor: {input_tensor}, label_tensor: {self.label_tensor}")

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            #  debug intermediate layer outputs.
            # print(f"[DEBUG] Layer output: {input_tensor}")

        loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        # print(f"[DEBUG] Loss computed: {loss}")

        return loss

    def backward(self):

        error_tensor = self.loss_layer.backward(self.label_tensor)
        # print(f"[DEBUG] Initial error tensor from loss layer: {error_tensor}")

        # Propagate the error tensor through the layers in reverse order.
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
            # print(f"[DEBUG] Error tensor after layer: {error_tensor}")

    def append_layer(self, layer):
        if hasattr(layer, 'trainable') and layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
            # print(f"[DEBUG] Initialized layer with weights and biases.")

        self.layers.append(layer)
        # print(f"[DEBUG] Layer appended: {layer}")

    def train(self, iterations):
        for iter_num in range(1, iterations + 1):
            loss = self.forward()
            # Store the loss for analysis.
            self.loss.append(loss)
            # print(f"[DEBUG] Iteration {iter_num}, Loss: {loss}")
            self.backward()

    def test(self, input_tensor):
        # Forward pass through all layers (excluding the loss layer).
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            # print(f"[DEBUG] Layer output during testing: {input_tensor}")

        return input_tensor
