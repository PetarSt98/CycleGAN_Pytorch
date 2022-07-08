import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CARLA_DIR = ''
DARWIN_DIR = ''
GEN_DIR = ''
GEN_DATASET_DIR = ''
BATCH_SIZE = 1
LAMBDA_IDENTITY = 1
LAMBDA_CYCLE = 1
NUM_WORKERS = 0
NUM_EPOCHS = 10
LEARNING_RATE = np.concatenate((np.linspace(0.0001, 0.0001, num=NUM_EPOCHS//3),
                                np.linspace(0.0001, 0, num=int((NUM_EPOCHS//3)*2))))
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_C = "genh.pth.tar"
CHECKPOINT_GEN_D = "genz.pth.tar"
CHECKPOINT_CRITIC_C = "critich.pth.tar"
CHECKPOINT_CRITIC_D = "criticz.pth.tar"
LOAD_CHECKPOINT_GEN_C = "genh.pth.tar"
LOAD_CHECKPOINT_GEN_D = "genz.pth.tar"
LOAD_CHECKPOINT_CRITIC_C = "critich.pth.tar"
LOAD_CHECKPOINT_CRITIC_D = "criticz.pth.tar"


transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)