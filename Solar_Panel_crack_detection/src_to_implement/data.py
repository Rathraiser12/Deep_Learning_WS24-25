from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

# Provided normalization values
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std  = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        """
        Args:
            data (pandas.DataFrame): DataFrame containing the image file paths and labels.
            mode (str): Either "train" or "val". Determines the transformation pipeline.
        """
        self.data = data
        self.mode = mode.lower()

        # Define transformations for train and validation modes.
        # Note: For training, we include a simple data augmentation (random horizontal flip).
        if self.mode == "train":
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.ColorJitter(brightness=0.2, contrast=0.2),
                #tv.transforms.RandomRotation(degrees=10),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:  # For validation or any other mode, we use a fixed transformation.
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Given an index, returns a tuple (image, label) where:
          - image is a torch.Tensor transformed according to self.transform.
          - label is a torch.Tensor containing the corresponding label.
        """
        # Get the row from the dataframe.
        row = self.data.iloc[index]

        # Read the image using skimage.io.imread.
        # Assumes that the dataframe has a column "filename" with the image path.
        img = imread(row["filename"])

        # Check if the image is grayscale. Many grayscale images have shape (H, W)
        # or shape (H, W, 1). In these cases, convert to RGB.
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            img = gray2rgb(img)

        # Apply the transformation pipeline.
        img = self.transform(img)

        # Extract the label.
        # Assumes the dataframe has a column "label". Adjust as needed.
        crack_label = row["crack"]
        inactive_label = row["inactive"]
        label = torch.tensor([crack_label, inactive_label], dtype=torch.float)

        return img, label
