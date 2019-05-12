from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

class WildAnimalDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, meta_path, photo_path, transform):
        """
        Args:
            meta_path (string): Path to the csv file with annotations.
            photo_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta = pd.io.excel.read_excel(meta_path)  
        self.photo_path = photo_path
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        img_name = os.path.join(self.photo_path, self.meta.Name[idx])
        image = Image.open(img_name)
        image_transformed = self.transform(image)
        image_resized = transforms.Resize((1000,1500))(image)
        return [image_transformed, np.array(image_resized)]