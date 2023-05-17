from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch
import cv2
import os
import PIL
import pandas as pd


class FacesDataset(Dataset):
    def __init__(self, root, image_dir, csv_file, transform):
        self.root = root
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_name = os.path.join(self.image_dir, self.image_files[index])
        image = PIL.Image.open(image_name)
        label = (self.dataframe.iloc[index]["age_group"],self.dataframe.iloc[index]["gender"])
        if self.transform:
            image = self.transform(image)
        return (image, label)