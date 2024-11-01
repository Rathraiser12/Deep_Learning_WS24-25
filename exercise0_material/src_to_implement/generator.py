import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import resize  # Import for resizing

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=True):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.current_epoch_num = 0
        self.index = 0

        print(f"Image directory path: {self.file_path}")
        print(f"Label file path: {self.label_path}")

        # Load labels from JSON and normalize keys
        self.labels = self._load_labels()

        # Preload images
        self.images = {}
        self.image_files = [f for f in os.listdir(self.file_path) if f.endswith('.npy')]
        print(f"Found {len(self.image_files)} .npy files in {self.file_path}.")

        for image_file in self.image_files:
            self.images[image_file] = self._load_image(image_file)

        # Class label dictionary
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                           5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    def _load_labels(self):
        # Load JSON labels and add `.npy` to any keys without it
        with open(self.label_path, 'r') as f:
            labels = json.load(f)

        normalized_labels = {}
        for key, value in labels.items():
            if not key.endswith('.npy'):
                key += '.npy'
            normalized_labels[key] = value

        return normalized_labels

    def _load_image(self, image_file):
        full_path = os.path.join(self.file_path, image_file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The image file {full_path} does not exist.")
        img = np.load(full_path)
        img = resize(img, self.image_size)  # Resize to target size
        img = img / 255.0  # Normalize to [0, 1]
        return np.round(img, decimals=5)

    def next(self):
        # Check if we need to reset the index for a new epoch
        if self.index + self.batch_size > len(self.image_files):
            self.index = 0
            self.current_epoch_num += 1
            if self.shuffle:
                random.shuffle(self.image_files)  # Shuffle at the start of each epoch

        batch_files = self.image_files[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        images, labels = [], []
        for image_file in batch_files:
            image = self.images[image_file]
            image = self.augment(image)
            images.append(image)
            labels.append(self.labels.get(image_file, -1))  # Use a default if label is missing

        return np.array(images), np.array(labels)

    def augment(self, img):
        if self.mirroring:
            if random.choice([True, False]):
                img = np.flip(img, axis=1)  # Horizontal flip
            if random.choice([True, False]):
                img = np.flip(img, axis=0)  # Vertical flip

        if self.rotation:
            k = random.choice([0, 1, 2, 3])  # Rotate by 0, 90, 180, or 270 degrees
            img = np.rot90(img, k)

        return img

    def current_epoch(self):
        return self.current_epoch_num

    def class_name(self, label):
        # Retrieve class name by label
        return self.class_dict.get(label, "Unknown")

    def show(self):
        images, labels = self.next()
        plt.figure(figsize=(10, 10))

        for i in range(min(self.batch_size, 16)):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i])
            plt.title(self.class_name(labels[i]))
            plt.axis('off')

        plt.show()
