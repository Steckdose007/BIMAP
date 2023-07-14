from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch
import cv2
import os
import PIL
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class FacesDataset(Dataset):
    def __init__(self, root, image_dir, label_list, transform):
        self.root = root
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.label_list = label_list
        self.transform = transform
        self.classes_list= ["0-2","3-6","7-9","10-14","15-19","20-29","30-39","40-49","50-69","70-120",]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        number,gender,age = self.label_list[index]
        image_name = os.path.join(self.image_dir, self.image_files[number])
        image =PIL.Image.open(image_name)
        label = next((i for i, item in enumerate(self.classes_list) if item == age), None)
        if label is None:
            raise ValueError("No matching value found in the list")
        if self.transform:
            image = self.transform(image)
        return (image, label)


