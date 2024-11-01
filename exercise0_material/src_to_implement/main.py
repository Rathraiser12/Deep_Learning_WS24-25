from pattern import Checker
from pattern import Circle
from pattern import Spectrum

def main():
    resolution = 300
    tile_size = 30

    checker = Checker(resolution, tile_size)
    checker.draw()
    checker.show()


    radius = 50
    position = (150, 150)
    circle = Circle(resolution, radius, position)
    circle.draw()
    circle.show()

    spectrum = Spectrum(resolution=256)
    spectrum.draw()
    spectrum.show()

if __name__ == "__main__":
    main()
