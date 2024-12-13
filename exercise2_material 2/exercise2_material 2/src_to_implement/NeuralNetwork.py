import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        """
        Initializes a neural network with the given optimizer, weights_initializer, and bias_initializer.

        :param optimizer: Optimizer object to be used for weight updates
        :param weights_initializer: Weights initializer to initialize layer weights
        :param bias_initializer: Bias initializer to initialize layer biases
        """
        self.optimizer = optimizer  # Optimizer object received upon construction
        self.loss = []  # List to store loss values per iteration
        self.layers = []  # List to hold the architecture (layers)
        self.data_layer = None  # To be set externally
        self.loss_layer = None  # To be set externally
        self.label_tensor = None  # To store label tensor from data layer
        self.weights_initializer = weights_initializer  # Store weights initializer
        self.bias_initializer = bias_initializer  # Store bias initializer

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
        # Starting backpropagation from the loss_layer
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # Propagate the error_tensor back through the layers in reverse order
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if hasattr(layer, 'trainable') and layer.trainable:
            # Make a deep copy of the optimizer


            # Pass the weights and bias initializers to the layer for initialization
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for iter_num in range(1, iterations + 1):
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
