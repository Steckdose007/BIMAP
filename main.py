import pandas as pd
from torch.utils.data import Dataset
from Diffusion import Diffusion
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import cv2
from Dataset import FacesDataset
from pathlib import Path
import os

# Use this to show a Image at index 0-70000
"""
Use this to show a Image at index 0-70000
"""
def show_image(image, label, dataset):
    print(f"Label: {label}")
    img = image.permute(1, 2, 0)
    img = (img.clamp(-1, 1) + 1) / 2
    img = (img * 255).type(torch.uint8)
    plt.imshow(img)
    plt.show()

"""
This method initializes the dataloader with the args defined below
"""
def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    root = Path(os.getcwd())
    image_dir = root / args.dataset_path
    csv_file = root / 'ffhq_aging_labels.csv'
    dset = FacesDataset(root, image_dir, csv_file, transform=transforms)
    #train_dataset, test_dataset = torch.utils.data.random_split(dset, [50, 18])
    #show_image(*dset[1], dset)
    dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=False)
    return dataloader

"""
This method diffusions the image. Here later the training will run.
"""
def test_noising(args):
    dataloader = get_data(args)
    diffusion = Diffusion(img_size=args.image_size, device=args.device)
    l = len(dataloader)
    print("Lenght: ",l)
    for idx, (data, target) in enumerate(dataloader):
        print('Batch index: ', idx)
        print('Batch size: ', data[0].size())
        print('Batch label: ', target[0])
        print("idx",idx)
        t = diffusion.sample_timesteps(data.shape[0])
        #print(t)
        x_t, noise = diffusion.noise_images(data, t)
        print(x_t.shape,noise.shape)
        num_images = 10
        T = 1000
        stepsize = int(T / num_images)
        fig, ax = plt.subplots(1,11)
        plt.axis('off')
        inter=0
        for idx in range(0, T, stepsize):
            t = diffusion.sample_timesteps(data.shape[0])
            #print(t)
            x_t, noise = diffusion.noise_images(data, [idx])
            img = x_t[1]
            img = img.permute(1, 2, 0)
            img = (img.clamp(-1, 1) + 1) / 2
            img = (img * 255).type(torch.uint8)
            # Take first image of batch
            ax[inter].imshow(img)
            inter +=1

        plt.show()
        break

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 513
    args.dataset_path = r"images1024x1024/"
    args.device = "cpu"
    args.lr = 3e-4
    test_noising(args)

if __name__ == "__main__":
    launch()

