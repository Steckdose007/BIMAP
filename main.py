from utils import *
from Diffusion import Diffusion
import torch.nn as nn
import torch
from torch import optim
import torchvision
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import cv2
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

"""
Use this to show a Image at index 0-70000
"""
def show_image(dataset):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    for i, image in enumerate(dataset):
        print(len(image))
        axs[i].imshow(image[i])

        axs[i].set_axis_off()

    fig.show()

"""
This method initializes the dataloader with the args defined below
"""
def get_data(args):
    transforms = torchvision.transforms.Compose([
        #torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    root = Path(os.getcwd())
    image_dir = root / args.dataset_path
    csv_file = root / 'ffhq_aging_labels.csv'
    dataframe = pd.read_csv(csv_file)
    image_number = dataframe["image_number"].tolist()
    gender = dataframe["gender"].tolist()
    age_group = dataframe["age_group"].tolist()
    label_list=[]
    for i in range(len(image_number)):
        label_list.append((image_number[i],gender[i],age_group[i]))
    x_train, x_test = train_test_split(label_list, test_size=0.1)
    print(len(x_test),len(x_train))
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

def show_diffusion(image, target,num_train_timesteps,device,diffusion):
    print(target)
    num_images = 10
    T = 1000
    stepsize = int(T / num_images)
    fig, ax = plt.subplots(1, 11)
    plt.axis('off')
    inter = 0
    for idx in range(0, T, stepsize):
        t = torch.tensor(idx)
        # t = sample_timesteps(num_train_timesteps, 1).to(device)
        noise = torch.randn(image.shape)
        x_t = diffusion.add_noise(image, noise, t)
        img = x_t
        img = img.permute(1, 2, 0)
        img = (img.clamp(-1, 1) + 1) / 2
        img = (img * 255).type(torch.uint8)
        # Take first image of batch
        ax[inter].imshow(img)
        inter += 1

    plt.show()

def tokenize_captions(captions,tokenizer):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

"""
Train Loop
"""
def train(args):
    setup_logging(args.run_name)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    dataloaders = get_data(args)
    device = args.device
    mse = nn.MSELoss()
    repo_id = "runwayml/stable-diffusion-v1-5"
    # pipeline = StableDiffusionPipeline.from_ckpt("./model_ckpt/v1-5-pruned.ckpt").to(device)
    # print(pipeline)
    #rom huggingface_hub import hf_hub_download
    #ckpt_path = hf_hub_download(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4-full-ema.ckpt", use_auth_token=True)
    model_id = "runwayml/stable-diffusion-v1-5"
    stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id, variant="non_ema")
    model = stable_diffusion_txt2img.unet
    #model_text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer"#, revision=args.revision The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so revision can be any identifier allowed by git.
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder"
    )
    # vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=args.revision)
    # unet = UNet2DConditionModel.from_pretrained(
    #     model_id, subfolder="unet"
    # )
    print("model loaded")
    #print(model)
    vgg = models.vgg16()
    summary(vgg, (3, 513, 513))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(img_size=args.image_size, device=args.device)
    num_train_timesteps = 1000
    diffusion = DDPMScheduler(num_train_timesteps)


    for epoch in range(args.epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloaders[phase])
            for idx, (data, target) in enumerate(pbar):
                images = data.to(device)
                print('Batch index: ', idx)
                print('Batch size: ', data.size())
                print('Batch label: ', np.shape(target), type(target))
                print("idx",idx)
                # show_diffusion(images[0], (target[0][0], target[1][0]), num_train_timesteps, device, diffusion)
                # break
                target_tokenized = tokenize_captions(target,tokenizer).to(device)
                encoder_hidden_states = text_encoder(target)[0]
                print(encoder_hidden_states)
                #print(target_tokenized)
                t = sample_timesteps(num_train_timesteps,data.shape[0]).to(device)
                noise = torch.randn(data.shape).float().to(device)
                x_t = diffusion.add_noise(images, noise, t)
                alpha_tensor = torch.ones(args.batch_size, 1, 513, 513).to(device)
                x_t = torch.cat((x_t, alpha_tensor), dim=1)
                print(target)
                print(np.shape(x_t),np.shape(target_tokenized))
                predicted_noise = model(x_t, t, target_tokenized)
                loss = mse(noise, predicted_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(), global_step=epoch + idx)

            # sampled_images = diffusion.sample(model, n=data.shape[0])
            # does this work with the hugginface?
            # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            # print(x_t.shape,noise.shape)


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 513
    args.dataset_path = r"images1024x1024/"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == "__main__":
    launch()

