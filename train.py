import torch
from dataset import CarlaDarwinDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import random

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def train_fn(disc_C, disc_D, gen_D, gen_C, loader, optimizer_G, optimizer_D_C, optimizer_D_D, cycle_loss, identity_loss,
             adversarial_loss, fake_A_buffer, fake_B_buffer):
    loop = tqdm(loader, leave=True)
    for idx, (Darwin, Carla) in enumerate(loop):
        real_image_D = Darwin.to(config.DEVICE)
        real_image_C = Carla.to(config.DEVICE)
        real_label = torch.full((config.BATCH_SIZE, 1, 1, 1), 1, device=config.DEVICE, dtype=torch.float32)
        fake_label = torch.full((config.BATCH_SIZE, 1, 1, 1), 0, device=config.DEVICE, dtype=torch.float32)
        # Train Discriminators C and D

        # Set G_C and G_D's gradients to zero
        optimizer_G.zero_grad()

        # Identity loss
        # G_D2C(C) should equal C if real C is fed
        identity_image_C = gen_C(real_image_C)
        loss_identity_C = identity_loss(identity_image_C, real_image_C)
        # G_C2D(D) should equal D if real C is fed
        identity_image_D = gen_D(real_image_D)
        loss_identity_D = identity_loss(identity_image_D, real_image_D)

        # GAN loss
        # GAN loss D_C(G_C(C))
        fake_image_C = gen_C(real_image_D)
        fake_output_C = disc_C(fake_image_C)
        loss_GAN_D2C = adversarial_loss(fake_output_C, real_label)
        # GAN loss D_D(G_D(D))
        fake_image_darwin = gen_D(real_image_C)
        fake_output_D = disc_D(fake_image_darwin)
        loss_GAN_C2D = adversarial_loss(fake_output_D, real_label)

        # Cycle loss
        recovered_image_C = gen_C(fake_image_darwin)
        loss_cycle_CDC = cycle_loss(recovered_image_C, real_image_C)

        recovered_image_D = gen_D(fake_image_C)
        loss_cycle_DCD = cycle_loss(recovered_image_D, real_image_D)

        # Combined loss and calculate gradients
        identity_loss_e = (loss_identity_C + loss_identity_D) / 2
        gan_loss = (loss_GAN_C2D + loss_GAN_D2C) / 2
        cycle_loss_e = (loss_cycle_CDC + loss_cycle_DCD) / 2
        errG = identity_loss_e * config.LAMBDA_IDENTITY + gan_loss * 10 + cycle_loss_e * config.LAMBDA_CYCLE

        # Calculate gradients for G_C and G_D
        errG.backward()
        # Update G_C and G_D's weights
        optimizer_G.step()

        # Set D_C gradients to zero
        optimizer_D_C.zero_grad()

        # Real C image loss
        real_output_C = disc_C(real_image_C)
        errD_real_C = adversarial_loss(real_output_C, real_label)

        # Fake C image loss
        fake_image_C = fake_A_buffer.push_and_pop(fake_image_C)
        fake_output_C = disc_C(fake_image_C.detach())
        errD_fake_C = adversarial_loss(fake_output_C, fake_label)

        # Combined loss and calculate gradients
        errD_C = (errD_real_C + errD_fake_C) / 2

        # Calculate gradients for D_C
        errD_C.backward()
        # Update D_C weights
        optimizer_D_C.step()

        # Set D_D gradients to zero
        optimizer_D_D.zero_grad()

        # Real D image loss
        real_output_D = disc_D(real_image_D)
        errD_real_D = adversarial_loss(real_output_D, real_label)

        # Fake D image loss
        fake_image_darwin = fake_B_buffer.push_and_pop(fake_image_darwin)
        fake_output_D = disc_D(fake_image_darwin.detach())
        errD_fake_D = adversarial_loss(fake_output_D, fake_label)

        # Combined loss and calculate gradients
        errD_D = (errD_real_D + errD_fake_D) / 2

        # Calculate gradients for D_D
        errD_D.backward()
        # Update D_D weights
        optimizer_D_D.step()

        disc_loss = (errD_D + errD_C) / 2

        if idx % 100 == 0:
            save_image(recovered_image_C * 0.5 + 0.5, f"{config.GEN_DIR}/reconstructed_carla/Recon_Carla_{idx}.png")
            save_image(recovered_image_D * 0.5 + 0.5, f"{config.GEN_DIR}/reconstructed_darwin/Recon_Darwin_{idx}.png")
            save_image(fake_image_C * 0.5 + 0.5, f"{config.GEN_DIR}/fake_carla/Fake_Carla_{idx}.png")
            save_image(fake_image_darwin * 0.5 + 0.5, f"{config.GEN_DIR}/fake_darwin/Fake_Darwin_{idx}.png")
        loop.set_postfix(Cycle_loss=cycle_loss_e.item(), Identity_loss=identity_loss_e.item(), Gan_loss=gan_loss.item(),
                         Disc_loss=disc_loss.item())


def train():
    disc_C = Discriminator(in_channels=3).to(config.DEVICE)
    disc_D = Discriminator(in_channels=3).to(config.DEVICE)
    gen_D = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_C.parameters()) + list(disc_D.parameters()),
        lr=config.LEARNING_RATE[0],
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_D.parameters()) + list(gen_C.parameters()),
        lr=config.LEARNING_RATE[0],
        betas=(0.5, 0.999),
    )

    cycle_loss = torch.nn.L1Loss().to(config.DEVICE)
    identity_loss = torch.nn.L1Loss().to(config.DEVICE)
    adversarial_loss = torch.nn.MSELoss().to(config.DEVICE)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.LOAD_CHECKPOINT_GEN_C, gen_C, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.LOAD_CHECKPOINT_GEN_D, gen_D, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.LOAD_CHECKPOINT_CRITIC_C, disc_C, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.LOAD_CHECKPOINT_CRITIC_D, disc_D, opt_disc, config.LEARNING_RATE,
        )

    dataset = CarlaDarwinDataset(
        root_carla=config.CARLA_DIR, root_darwin=config.DARWIN_DIR, transform=config.transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    for epoch in range(config.NUM_EPOCHS):
        optimizer_G = torch.optim.Adam(list(gen_D.parameters()) + list(gen_C.parameters()),
                                       lr=config.LEARNING_RATE[epoch], betas=(0.5, 0.999))
        optimizer_D_C = torch.optim.Adam(disc_C.parameters(), lr=config.LEARNING_RATE[epoch], betas=(0.5, 0.999))
        optimizer_D_D = torch.optim.Adam(disc_D.parameters(), lr=config.LEARNING_RATE[epoch], betas=(0.5, 0.999))

        train_fn(disc_C, disc_D, gen_D, gen_C, loader, optimizer_G, optimizer_D_C, optimizer_D_D, cycle_loss,
                 identity_loss, adversarial_loss, fake_A_buffer, fake_B_buffer)

        if config.SAVE_MODEL:
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_GEN_C)
            save_checkpoint(gen_D, opt_gen, filename=config.CHECKPOINT_GEN_D)
            save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_CRITIC_D)
            save_checkpoint(disc_D, opt_disc, filename=config.CHECKPOINT_CRITIC_C)

