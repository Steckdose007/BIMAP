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
from diffusers import UNet2DConditionModel
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
from transformers import CLIPProcessor, CLIPTextModel
import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import PIL
#model = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", variant="non-ema")
# from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
#
# model_id = "runwayml/stable-diffusion-v1-5"
# stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id, variant="non_ema")
# stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
#     vae=stable_diffusion_txt2img.vae,
#     text_encoder=stable_diffusion_txt2img.text_encoder,
#     tokenizer=stable_diffusion_txt2img.tokenizer,
#     unet=stable_diffusion_txt2img.unet,
#     scheduler=stable_diffusion_txt2img.scheduler,
#     safety_checker=None,
#     feature_extractor=None,
#     requires_safety_checker=False,
# )
# components = stable_diffusion_img2img.components
# print(stable_diffusion_txt2img.unet)
"""
IDEE: Train normal text to img and use img as gaussian noise
"""
device = "cuda"

model_id_or_path = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)

pipe = pipe.to(device)
url = "C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/images1024x1024/00012.png"

init_image = PIL.Image.open(url).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "Make the Man look 80 Years old"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

images[0].save("fantasy_landscape.png")