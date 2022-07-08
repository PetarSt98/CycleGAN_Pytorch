from utils import load_checkpoint
from torch.utils.data import DataLoader
import torch.optim as optim
import config
from dataset import GenDataset
from tqdm import tqdm
from torchvision.utils import save_image
from generator_model import Generator


def gen_fn(idx, gen_D, gen_C, image):
    image_D = gen_D(image)
    image_C = gen_C(image)
    save_image(image_D * 0.5 + 0.5, f"{config.GEN_DIR}/generated_carla/Gen_Carla_{idx}.png")
    save_image(image_C * 0.5 + 0.5, f"{config.GEN_DIR}/generated_darwin/Gen_Darwin_{idx}.png")


def generate():
    gen_D = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_D.parameters()) + list(gen_C.parameters()),
        lr=config.LEARNING_RATE[0],
        betas=(0.5, 0.999),
    )

    load_checkpoint(
        config.LOAD_CHECKPOINT_GEN_C, gen_C, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.LOAD_CHECKPOINT_GEN_D, gen_D, opt_gen, config.LEARNING_RATE,
    )

    dataset = GenDataset(root_gen=config.GEN_DATASET_DIR, transform=config.transforms)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    loop = tqdm(loader, leave=True)
    for idx, image in enumerate(loop):
        gen_fn(idx, gen_D, gen_C, image)
