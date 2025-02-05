import copy
import pickle

def store_network(filename_path, network_obj):
    # Debug: Serializing network
    pickle.dump(network_obj, open(filename_path, 'wb'))

def load_network(filename_path, data_source):
    # Debug: Loading network from file
    restored_net = pickle.load(open(filename_path, 'rb'))
    restored_net.data_layer = data_source
    return restored_net

class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        # Debug: Initializing NeuralNetwork
        self.optimizer = optimizer
        self.loss = []  # Accumulate loss values each iteration
        self.layers = []  # List of layers in the network
        self.loss_layer = None  # Final loss function
        self.phaseval = None  # Will store the current phase (train/test)

        # Make deep copies of initializers
        self.weights_initializer = copy.deepcopy(weights_initializer)
        self.bias_initializer = copy.deepcopy(bias_initializer)

        # Data and labels
        self.data_layer = None
        self.label_tensor = None

    def __getstate__(self):
        saved_state = self.__dict__.copy()
        del saved_state['data_layer']
        return saved_state

    def __setstate__(self, saved_state):
        self.__dict__ = saved_state

    def forward(self):
        # Debug: Starting forward pass
        input_data, self.label_tensor = self.data_layer.next()
        reg_accum = 0

        for layer in self.layers:
            layer.testing_phase = False
            input_data = layer.forward(input_data)

            # Include regularization if defined and layer is trainable
            if self.optimizer.regularizer is not None and layer.trainable:
                reg_accum += self.optimizer.regularizer.norm(layer.weights)

        # Return total loss (network loss + regularization)
        return self.loss_layer.forward(input_data, self.label_tensor) + reg_accum

    def backward(self):
        # Debug: Starting backward pass
        backprop_err = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            backprop_err = layer.backward(backprop_err)

    def append_layer(self, layer):
        # Debug: Appending layer to the network
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, num_iter):
        self.phase = False  # Set to training mode
        for _ in range(num_iter):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    @property
    def phase(self):
        return self.layers[0].testing_phase if self.layers else False

    @phase.setter
    def phase(self, new_mode):
        # Debug: Setting phase for all layers
        for layer in self.layers:
            layer.testing_phase = new_mode
