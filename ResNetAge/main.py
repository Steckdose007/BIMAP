import torch.nn as nn
import torch
from torch import optim
import torchvision
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import cv2
import logging
import pandas as pd
from diffusers import DDPMScheduler
from diffusers import UNet2DConditionModel,AutoencoderKL
from Dataset import FacesDataset
from pathlib import Path
import os
from diffusers import StableDiffusionPipeline
import logging
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from timm import create_model
from sklearn.model_selection import train_test_split
from torchvision import models
from torchsummary import summary
from transformers import CLIPProcessor, CLIPTextModel, CLIPTokenizer
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    root = Path(os.getcwd())
    image_dir = args.dataset_path
    csv_file = root / 'ffhq_aging_labels.csv'
    dataframe = pd.read_csv(csv_file)
    image_number = dataframe["image_number"].tolist()
    gender = dataframe["gender"].tolist()
    age_group = dataframe["age_group"].tolist()
    label_list=[]
    for i in range(len(image_number)):
        label_list.append((image_number[i],gender[i],age_group[i]))
    x_train, x_test = train_test_split(label_list, test_size=0.1)
    print("lenght_list",len(x_test),len(x_train))
    dset_train = FacesDataset(root, image_dir, x_train, transform=transforms)
    dset_x_test = FacesDataset(root, image_dir, x_test, transform=transforms)
    dataloaders = {
        'train': DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(dset_x_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
    }
    l = len(dataloaders["train"])
    print("Lenght train: ",l)
    l = len(dataloaders["val"])
    print("Lenght val: ", l)
    return dataloaders

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def train(args):
    setup_logging(args.run_name)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    dataloaders = get_data(args)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print("device",device)
    model = models.resnet50(pretrained=True)
    #print(model)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloaders["train"])
        idx=0
        for inputs, labels in pbar:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders["val"]:
                        inputs, labels = inputs.to(device),labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch + idx)
            idx +=1
        train_losses.append(running_loss / len(dataloaders["train"]))
        test_losses.append(test_loss / len(dataloaders["val"]))
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Train loss: {running_loss / print_every:.3f}.. "
              f"Test loss: {test_loss / len(dataloaders['val']):.3f}.. "
              f"Test accuracy: {accuracy / len(dataloaders['val']):.3f}")
        running_loss = 0
        model.train()
    torch.save(model, 'aerialmodel.pth')
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()





def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Uncondtional"
    args.epochs = 1
    args.batch_size = 32
    args.image_size = 80
    args.dataset_path = r"C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/images1024x1024/"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == "__main__":
    launch()