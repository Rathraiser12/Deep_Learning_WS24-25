from pattern import Checker
from pattern import Circle
from pattern import Spectrum

if __name__ == "__main__":
    checker = Checker(resolution=256, tile_size=32)
    checker.draw()
    checker.show()

    circle = Circle(resolution=256, radius=50, position=(128, 128))
    circle.draw()
    circle.show()

    spectrum = Spectrum(resolution=256)
    spectrum.draw()
    spectrum.show()

