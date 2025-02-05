import sys

# Debug: Checking current system path
# print(sys.path)  #

from Layers import Conv, FullyConnected, Flatten, ReLU, Pooling, SoftMax, Initializers
from Optimization import Loss, Constraints, Optimizers
import NeuralNetwork as nn

def build():
    # Debug: Creating and configuring Adam optimizer
    adam_instance = Optimizers.Adam(5e-4, 0.9, 0.999)
    adam_instance.add_regularizer(Constraints.L2_Regularizer(4e-4))

    # Debug: Initializing NeuralNetwork with He initializers
    model = nn.NeuralNetwork(adam_instance, Initializers.He(), Initializers.He())

    # Debug: Appending layers to the model
    model.append_layer(Conv.Conv((1, 1), (1, 5, 5), 6))
    model.append_layer(ReLU.ReLU())
    model.append_layer(Pooling.Pooling((2, 2), (2, 2)))

    model.append_layer(Conv.Conv((1, 1), (6, 5, 5), 16))
    model.append_layer(ReLU.ReLU())
    model.append_layer(Pooling.Pooling((2, 2), (2, 2)))

    model.append_layer(Flatten.Flatten())

    model.append_layer(FullyConnected.FullyConnected(16 * 7 * 7, 120))
    model.append_layer(ReLU.ReLU())

    model.append_layer(FullyConnected.FullyConnected(120, 84))
    model.append_layer(ReLU.ReLU())

    model.append_layer(FullyConnected.FullyConnected(84, 10))
    model.append_layer(ReLU.ReLU())

    model.append_layer(SoftMax.SoftMax())

    # Debug: Setting loss layer
    model.loss_layer = Loss.CrossEntropyLoss()

    # Debug: Build complete, returning model
    return model
