import pandas as pd
from PIL import Image
import numpy as np
from datasets import IterableDataset
import os
from matplotlib import pyplot as plt
import torch
from diffusers import DDPMScheduler
import torchvision
import random
def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class ImageDataset(IterableDataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.image_files = os.listdir(img_dir)
        self.img_dir = img_dir
        self.column_names = ["input_image", "edit_prompt", "edited_image"]
        self.noise_steps = 1000
        self.diffusion = DDPMScheduler(self.noise_steps)
        self.device = "cpu"
        self.transforms = torchvision.transforms.Compose([
        #torchvision.transforms.ToPILImage(),
        #torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(513, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    def __iter__(self):
        for idx, row in self.df.iterrows():
            t = random.randint(0, 300)
            t = torch.tensor(t)
            age = row['age_group']
            gender = row['gender']
            label =  "Make this {} look {} years old.".format(gender,age)
            #load image
            image_path = os.path.join(self.img_dir, self.image_files[idx])
            img = Image.open(image_path).convert('RGB')
            img = img.resize((513, 513))
            image_array = np.array(img)
            img = self.transforms(img)#.to(self.device)

            noise = torch.randn(img.shape)
            x_t = self.diffusion.add_noise(img, noise, t)
            x_t = x_t.permute(1, 2, 0)
            x_t = (x_t.clamp(-1, 1) + 1) / 2
            x_t = (x_t * 255).type(torch.uint8)#.to("cpu")
            x_t = x_t.numpy()
            yield {'input_image': x_t, 'edit_prompt': label, 'edited_image': image_array}

    def column_names(self) :
        return self.column_names

    def __len__(self):
        return len(self.df)

    def load_dataset(file_path):
        dataset = ImageDataset('C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/ffhq_aging_labels.csv', 'C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/images1024x1024')
        return dataset

def get_item_1(dataset, index):
    for i, item in enumerate(dataset):
        if i == index:
            return item

if __name__ == "__main__":
    dataset = ImageDataset('C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/ffhq_aging_labels.csv', 'C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/images1024x1024')
    column_names = dataset.column_names
    print(column_names)
    item = get_item_1(dataset, 1)
    print(item['edited_image'].shape)
    print(item['input_image'].shape)
    print(item['edit_prompt'])
    plt.imshow(item['edited_image'], interpolation='nearest')
    plt.show()
    plt.imshow(item['input_image'], interpolation='nearest')
    plt.show()