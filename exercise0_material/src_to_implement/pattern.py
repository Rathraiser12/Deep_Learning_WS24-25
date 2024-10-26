import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        if self.resolution % (2 * self.tile_size) != 0:
            raise ValueError("Resolution must be divisible by 2 * tile_size.")

        pattern_tile = np.array([[0, 1], [1, 0]])  # Basic 2x2 tile
        tile = np.tile(pattern_tile, (self.tile_size, self.tile_size))
        self.output = np.tile(tile, (self.resolution // (2 * self.tile_size), self.resolution // (2 * self.tile_size)))
        return self.output

    def show(self):
        if self.output is None:
            raise ValueError("Draw the pattern before displaying it.")
        plt.imshow(self.output, cmap="gray")
        plt.axis("off")
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None


    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        xx, yy = np.meshgrid(x, y)
        distance = np.sqrt((xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2)
        self.output = (distance <= self.radius).astype(int)
        return self.output

    def show(self):
        if self.output is None:
            raise ValueError("Draw the pattern before displaying it.")
        plt.imshow(self.output, cmap="gray")
        plt.axis("off")
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        red_channel = np.linspace(0, 1, self.resolution).reshape(1, -1).repeat(self.resolution, axis=0)
        green_channel = np.linspace(0, 1, self.resolution).reshape(-1, 1).repeat(self.resolution, axis=1)
        blue_channel = np.ones((self.resolution, self.resolution)) - red_channel
        self.output = np.stack((red_channel, green_channel, blue_channel), axis=-1)
        return self.output

    def show(self):
        if self.output is None:
            raise ValueError("Draw the pattern before displaying it.")
        plt.imshow(self.output)
        plt.axis("off")
        plt.show()

