import os
from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator

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

    # paths fro training data
    image_dir = "/home/rathan/Pycharm_dl/github/DL24/exercise0_material/src_to_implement/data/exercise_data"  # Replace with your directory path containing .npy files
    label_file = "/home/rathan/Pycharm_dl/github/DL24/exercise0_material/src_to_implement/data/Labels.json"      # Replace with the full path to your label .npy file


    if not os.path.exists(image_dir):
        print(f"Image directory does not exist: {image_dir}")
        return
    if not os.path.exists(label_file):
        print(f"Label file does not exist: {label_file}")
        return

    # Initializing the ImageGenerator parameters to see the results
    batch_size = 10
    image_shape = [32, 32, 3]  # Example shape, adjust as necessary

    generator = ImageGenerator(
        file_path=image_dir,
        label_path=label_file,
        batch_size=batch_size,
        image_size=image_shape,
        rotation=False,
        mirroring=False,
        shuffle=False
    )
    generator.show()
    # DEBUG to check batch retrieval
    try:
        batch_images, batch_labels = generator.next()
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch labels: {batch_labels}")
    except Exception as e:
        print(f"Error retrieving batch: {e}")

if __name__ == "__main__":
    main()
