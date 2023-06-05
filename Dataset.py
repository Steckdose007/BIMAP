from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch
import cv2
import os
import PIL
import pandas as pd


class FacesDataset(Dataset):
    def __init__(self, root, image_dir, label_list, transform):
        self.root = root
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        number,gender,age = self.label_list[index]
        image_name = os.path.join(self.image_dir, self.image_files[number])
        image =PIL.Image.open(image_name)
        #image = cv2.imread(image_name,  cv2.IMREAD_UNCHANGED)
        label = (age + " "+gender)
        if self.transform:
            image = self.transform(image)
        return (image, label)