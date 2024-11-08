import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import resize  # Import for resizing

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size,
                 rotation=False, mirroring=False, shuffle=True):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size  # format: [height, width, channels]
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.current_epoch_num = 0
        self.index = 0

        print(f"Image directory path: {self.file_path}")
        print(f"Label file path: {self.label_path}")


        self.labels = self._load_labels()

        # Preloading images
        self.images = {}
        self.image_files = [f for f in os.listdir(self.file_path) if f.endswith('.npy')]

        # Shuffle image based on shuffle condition
        if self.shuffle:
            random.shuffle(self.image_files)

        print(f"Found {len(self.image_files)} .npy files in {self.file_path}.")

        for image_file in self.image_files:
            self.images[image_file] = self._load_image(image_file)

        # Class labels
        self.class_dict = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }

    def _load_labels(self):
        # Load labels and confirm the extension
        with open(self.label_path, 'r') as f:
            labels = json.load(f)

        normalized_labels = {}
        for key, value in labels.items():
            if not key.endswith('.npy'):
                key += '.npy'
            normalized_labels[key] = int(value)  # Ensuring labels are integers

        return normalized_labels

    def _load_image(self, image_file):
        full_path = os.path.join(self.file_path, image_file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The image file {full_path} does not exist.")
        img = np.load(full_path)

        # Debug: check the data type and min/max before processing
        # print(f"Loaded image {image_file} - dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

        # Converting to float32 for processing
        img = img.astype(np.float32)

        # DEBUG:Normalize if necessary
        if img.max() > 1.0:
            img /= 255.0

        # Ensuring Normalization
        img = np.clip(img, 0.0, 1.0)

        # Resize the image
        img = resize(img, self.image_size, anti_aliasing=True)

        # Debug: Print min and max after resizing
        # print(f"After resizing - min: {img.min()}, max: {img.max()}")

        return np.round(img, decimals=5)

    def next(self):
        images, labels = [], []
        batch_files = []
        while len(batch_files) < self.batch_size:
            remaining = len(self.image_files) - self.index
            needed = self.batch_size - len(batch_files)
            if remaining >= needed:
                batch_files.extend(self.image_files[self.index:self.index + needed])
                self.index += needed
            else:
                batch_files.extend(self.image_files[self.index:])
                self.index = 0
                self.current_epoch_num += 1
                if self.shuffle:
                    random.shuffle(self.image_files)
                needed = self.batch_size - len(batch_files)
                if needed > 0:
                    batch_files.extend(self.image_files[self.index:self.index + needed])
                    self.index += needed

        for image_file in batch_files:
            image = self.images[image_file]
            image = self.augment(image)
            images.append(image)
            labels.append(int(self.labels.get(image_file, -1)))  # Ensure labels are integers

        return np.array(images), np.array(labels)

    def augment(self, img):
        if self.mirroring:
            # Randomly choose one mirroring option
            mirror_type = random.choice(['horizontal', 'vertical', 'both'])
            if mirror_type == 'horizontal':
                img = np.flip(img, axis=1)  # Horizontal flip
            elif mirror_type == 'vertical':
                img = np.flip(img, axis=0)  # Vertical flip
            elif mirror_type == 'both':
                img = np.flip(img, axis=0)  # Vertical flip
                img = np.flip(img, axis=1)  # Horizontal flip

        if self.rotation:
            # Rotate by 90, 180, or 270 degrees
            k = random.choice([1, 2, 3])  # Excluding 0 to ensure rotation occurs
            img = np.rot90(img, k)

        # Debug: Check for invalid data after augmentation
        # if np.isnan(img).any() or np.isinf(img).any():
        #     print("Invalid image data after augmentation.")

        return img

    def current_epoch(self):
        return self.current_epoch_num

    def class_name(self, label):
        # Retrieve class name by label
        return self.class_dict.get(label, "Unknown")

    def show(self):
        images, labels = self.next()

        # Debug: Print min and max of the first image
        # print(f"First image - min: {images[0].min()}, max: {images[0].max()}")

        plt.figure(figsize=(10, 10))

        for i in range(min(self.batch_size, 16)):
            plt.subplot(4, 4, i + 1)
            img = images[i]
            # If the image has only one channel, reshape it appropriately
            if img.ndim == 2 or img.shape[2] == 1:
                plt.imshow(img.squeeze(), cmap='gray')
            else:
                plt.imshow(img)
            plt.title(self.class_name(labels[i]))
            plt.axis('off')

        plt.show()
