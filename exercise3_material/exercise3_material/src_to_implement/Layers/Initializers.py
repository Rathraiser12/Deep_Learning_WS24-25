import numpy as np


class Constant:

    def __init__(self, init_val=0.1):
        # Debug: Setting up Constant initializer with init_val
        self.init_val = init_val

    def initialize(self, shape_of_weights, input_dim, output_dim):
        # Debug: Constant init with shape={} input_dim={} output_dim={}
        # print(shape_of_weights, input_dim, output_dim)

        return np.full(shape_of_weights, self.init_val)


class UniformRandom:

    def initialize(self, shape_of_weights, input_dim, output_dim):
        # Debug: Uniform random initialization
        return np.random.uniform(low=0.0, high=1.0, size=shape_of_weights)


class Xavier:

    def initialize(self, shape_of_weights, input_dim, output_dim):
        std_dev = np.sqrt(2.0 / (input_dim + output_dim))

        # Debug: Using std_dev for Xavier initialization
        return np.random.normal(loc=0.0, scale=std_dev, size=shape_of_weights)


class He:
    def initialize(self, shape_of_weights, input_dim, output_dim):
        # Debug: He initialization in progress
        std_dev = np.sqrt(2.0 / input_dim)
        # print("He init std:", std_dev)

        return np.random.normal(loc=0.0, scale=std_dev, size=shape_of_weights)
