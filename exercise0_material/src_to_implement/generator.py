'''
import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        # Initialize properties
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        # Load labels
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)

        # Load images
        self.images = sorted(os.listdir(self.file_path))

        # Shuffle images if the flag is set
        if self.shuffle:
            np.random.shuffle(self.images)

        # Initialize indices for batching
        self.index = 0
        self.epoch = 0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # Calculate batch end index
        end = min(self.index + self.batch_size, len(self.images))
        batch_images = []
        batch_labels = []

        # Process each image in the batch
        for i in range(self.index, end):
            img_path = os.path.join(self.file_path, self.images[i])
            img = np.load(img_path)  # Load image

            # Resize and augment image
            img = self.resize(img)
            img = self.augment(img) if (self.rotation or self.mirroring) else img

            # Append image and label to batch
            batch_images.append(img)
            batch_labels.append(self.labels[self.images[i]])

        # Convert to numpy arrays
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        # Update index
        self.index = end
        if self.index >= len(self.images):
            self.index = 0
            self.epoch += 1
            if self.shuffle:
                np.random.shuffle(self.images)

        return batch_images, batch_labels
        #return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        if self.mirroring and np.random.rand() > 0.5:
            img = np.fliplr(img)  # Horizontal flip

        if self.rotation:
            rotations = np.random.choice([0, 1, 2, 3])  # 0°, 90°, 180°, 270°
            img = np.rot90(img, rotations)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict.get(x, "Unknown")

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        # Get a batch of images and labels
        batch_images, batch_labels = self.next()

        # Plot batch images
        plt.figure(figsize=(10, 10))
        for i in range(len(batch_images)):
            plt.subplot(1, self.batch_size, i + 1)
            plt.imshow(batch_images[i])
            plt.title(self.class_name(batch_labels[i]))
            plt.axis('off')
        plt.show()'''


import os
import numpy as np
import json

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Initialize properties
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        # Load labels from JSON file
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)

        # Load numpy file names
        self.images = [img for img in sorted(os.listdir(self.file_path)) if img.endswith('.npy')]

        # Shuffle images if the flag is set
        if self.shuffle:
            np.random.shuffle(self.images)

        # Initialize batching indices
        self.index = 0
        self.epoch = 0

    def next(self):
        # Calculate batch end index
        end = min(self.index + self.batch_size, len(self.images))
        batch_images = []
        batch_labels = []

        # Load numpy arrays in the batch
        for i in range(self.index, end):
            img_path = os.path.join(self.file_path, self.images[i])
            img = np.load(img_path)  # Load numpy array
            img = self.augment(img) if (self.rotation or self.mirroring) else img

            batch_images.append(img)
            batch_labels.append(self.labels[self.images[i]])

        # Convert to numpy arrays
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        # Update index for next batch
        self.index = end
        if self.index >= len(self.images):
            self.index = 0
            self.epoch += 1
            if self.shuffle:
                np.random.shuffle(self.images)

        return batch_images, batch_labels

    def augment(self, img):
        # Placeholder augmentation: apply mirroring and rotation if specified
        if self.mirroring:
            img = np.flip(img, axis=1)  # Horizontal flip
        if self.rotation:
            img = np.rot90(img)  # 90 degree rotation
        return img

    def current_epoch(self):
        return self.epoch

    def class_name(self, label_index):
        # Get class name from class dictionary
        return self.class_dict.get(label_index, "Unknown")

    def show(self):
        # Display the first batch for verification (requires calling next() before show())
        batch_images, batch_labels = self.next()
        print("Batch Images Shape:", batch_images.shape)
        print("Batch Labels:", batch_labels)
