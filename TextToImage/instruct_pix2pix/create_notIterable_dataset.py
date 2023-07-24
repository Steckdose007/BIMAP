import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
import os
from matplotlib import pyplot as plt
import torch
from diffusers import DDPMScheduler
import torchvision
import random
from huggingface_hub import login
import copy
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import matplotlib.pyplot as plt

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_dir):
        self.path = img_dir
        self.df = pd.read_csv(csv_path)
        self.image_files = os.listdir(img_dir)
        self.img_dir = img_dir
        self.column_names = ["original_image", "edit_prompt", "edited_image"]
        # self.noise_steps = 1000
        # self.diffusion = DDPMScheduler(self.noise_steps)
        #self.device = "cpu"
        # self.transforms = torchvision.transforms.Compose([
        # #torchvision.transforms.ToPILImage(),
        # #torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        # torchvision.transforms.RandomResizedCrop(513, scale=(0.8, 1.0)),
        # torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __getitem__(self,i):
        row = self.df.iloc[i]
        # t = random.randint(0, 300)
        # t = torch.tensor(t)
        age = row['age_group']
        gender = row['gender']
        label =  "Make this {} look {} years old.".format(gender,age)
        #load image
        image_path = os.path.join(self.img_dir, self.image_files[i])
        img = Image.open(image_path).convert('RGB')
        img = img.resize((513, 513))
        image_array = np.array(img)
        # img = self.transforms(img)#.to(self.device)
        #
        # noise = torch.randn(img.shape)
        # x_t = self.diffusion.add_noise(img, noise, t)
        # x_t = x_t.permute(1, 2, 0)
        # x_t = (x_t.clamp(-1, 1) + 1) / 2
        # x_t = (x_t * 255).type(torch.uint8)#.to("cpu")
        # x_t = x_t.numpy()
        return dict(original_image= Image.fromarray(image_array), edit_prompt= label, edited_image= Image.fromarray(image_array))

    def __len__(self):
        return len(self.df)


def preprocess_train(examples):
    print("hello")
    i=0
    for image in examples["original_image"]:
        image = torch.tensor(image)
        print(i)
        i+=1
    for image in examples["edited_image"]:
        image = torch.tensor(image)
        print(i)
        i+=1
    return examples

def gen_examples(dataset):
    def fn():
        i=0
        for sample in dataset:
            yield {
                "original_image": sample["original_image"],
                "edit_prompt": sample["edit_prompt"],
                "edited_image": sample["edited_image"],
            }

    return fn

if __name__ == "__main__":

    dataset_torch = ImageDataset('C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/ffhq_aging_labels.csv', 'C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/images1024x1024')
    print(dataset_torch[0])
    generator_fn = gen_examples(dataset_torch)
    print("Creating dataset...")
    dataset_test = Dataset.from_generator(
        generator_fn,
        features=Features(
            original_image=ImageFeature(),
            edit_prompt=Value("string"),
            edited_image=ImageFeature(),
        ),
    )

    # print(dataset_test.column_names)
    dataset_test.save_to_disk('dataset_hugg_big_non_diff')
    # dataset_name = "lol_ok"
    print("finished")
    #login(token= "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")
    #dataset_test= load_dataset("fusing/instructpix2pix-1000-samples")
    # column_names2 = dataset_test.column_names
    # print(column_names2)
    img = dataset_test[2]
    plt.imshow(img['original_image'], interpolation='nearest')
    plt.show()
    # # print(img)
    # l = len(dataset_test)
    # print("Lenght dataset: ", l)
    # dataset_test.with_transform(preprocess_train)
    # train_dataset = dataset_test.with_transform(preprocess_train)
    # l = len(train_dataset)
    # print("Lenght train: ", l)
    #
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     batch_size=10,
    #     num_workers=1,
    # )
    #
    # print("len dataloader",len(train_dataloader))
    # for step, batch in enumerate(train_dataloader):
    #     print(batch)
    #     break
