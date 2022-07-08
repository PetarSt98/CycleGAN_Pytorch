from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class CarlaDarwinDataset(Dataset):
    def __init__(self, root_darwin, root_carla, transform=None):
        self.root_darwin = root_darwin
        self.root_carla = root_carla
        self.transform = transform

        self.Darwin_images = os.listdir(root_darwin)
        self.Carla_images = os.listdir(root_carla)
        self.length_dataset = max(len(self.Darwin_images), len(self.Carla_images)) # 1000, 1500
        self.Darwin_len = len(self.Darwin_images)
        self.Carla_len = len(self.Carla_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        Darwin_img = self.Darwin_images[index % self.Darwin_len]
        Carla_img = self.Carla_images[index % self.Carla_len]

        Darwin_path = os.path.join(self.root_darwin, Darwin_img)
        Carla_path = os.path.join(self.root_carla, Carla_img)

        Darwin_img = np.array(Image.open(Darwin_path).convert("RGB"))
        Carla_img = np.array(Image.open(Carla_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=Darwin_img, image0=Carla_img)
            Darwin_img = augmentations["image"]
            Carla_img = augmentations["image0"]

        return Darwin_img, Carla_img


class GenDataset(Dataset):
    def __init__(self, root_gen, transform=None):
        self.root_gen = root_gen
        self.transform = transform

        self.Gen_images = os.listdir(root_gen)
        self.length_dataset = max(len(self.Gen_images), len(self.Gen_images)) # 1000, 1500
        self.Gen_len = len(self.Gen_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        Gen_img = self.Gen_images[index % self.Gen_len]
        Gen_path = os.path.join(self.root_gen, Gen_img)
        Gen_img = np.array(Image.open(Gen_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=Gen_img)
            Gen_img = augmentations["image"]

        return Gen_img





