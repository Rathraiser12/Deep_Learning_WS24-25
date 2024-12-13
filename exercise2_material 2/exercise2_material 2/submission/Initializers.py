import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value, dtype=np.float32)


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0, 1, size=weights_shape).astype(np.float32)


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        scale = np.sqrt(2) / np.sqrt(fan_in + fan_out)
        print(f"Xavier Init -> Normal Distribution: scale={scale}, fan_in={fan_in}, fan_out={fan_out}")
        # DEBUG: Weights are drawn from a normal distribution with zero mean and variance = scale^2.

        # weights = np.random.normal(0, scale, size=weights_shape).astype(np.float32)
        # print("Generated Weights:", weights)
        # return weights

        return np.random.normal(0, scale, size=weights_shape).astype(np.float32)


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        stddev = np.sqrt(2 / fan_in)

        # if fan_in == 0:
        #     stddev = 0.01  # fallback strategy

        return np.random.normal(0, stddev, size=weights_shape).astype(np.float32)
