# NeuralNetwork.py

import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer  # Optimizer object received upon construction
        self.loss = []  # List to store loss values per iteration
        self.layers = []  # List to hold the architecture (layers)
        self.data_layer = None  # To be set externally
        self.loss_layer = None  # To be set externally
        self.label_tensor = None  # To store label tensor from data layer

    def forward(self):
        # Get input_tensor and label_tensor from data_layer
        input_tensor, self.label_tensor = self.data_layer.next()

        # Pass the input_tensor through all layers except the loss_layer
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        # Pass the output through the loss_layer, which computes the loss
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss

    def backward(self):
        # Start backpropagation from the loss_layer
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # Propagate the error_tensor back through the layers in reverse order
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if hasattr(layer, 'trainable') and layer.trainable:
            # Make a deep copy of the optimizer
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            # Forward pass
            loss = self.forward()
            # Store the loss
            self.loss.append(loss)
            # Backward pass
            self.backward()

    def test(self, input_tensor):
        # Pass input_tensor through all layers except the loss_layer
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        # Return the output of the last layer
        prediction = input_tensor
        return prediction
