import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):

        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution must be divisible by 2 * tile_size.")

        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):

        # Creating a single tile with black (0) and white (1) squares
        single_tile = np.zeros((self.tile_size * 2, self.tile_size * 2), dtype=int)
        single_tile[:self.tile_size, :self.tile_size] = 0  # top-left black
        single_tile[:self.tile_size, self.tile_size:] = 1  # top-right white
        single_tile[self.tile_size:, :self.tile_size] = 1  # bottom-left white
        single_tile[self.tile_size:, self.tile_size:] = 0  # bottom-right black

        # Repeating the single tile pattern to match the resolution
        pattern = np.tile(single_tile, (self.resolution // (2 * self.tile_size), self.resolution // (2 * self.tile_size)))


        self.output = pattern
        return self.output.copy() # Return a copy to ensure independence from self.output

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
        # meshgrid of coordinates
        x, y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))

        # Calculating the distance from the center for each point
        dist_from_center = (x - self.position[0])**2 + (y - self.position[1])**2

        # Creating a binary circle pattern using boolean array
        self.output = dist_from_center <= self.radius**2
        return np.array(self.output, copy=True)

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
        return self.output.copy()

    def show(self):
        if self.output is None:
            raise ValueError("Draw the pattern before displaying it.")
        plt.imshow(self.output)
        plt.axis("off")
        plt.show()

