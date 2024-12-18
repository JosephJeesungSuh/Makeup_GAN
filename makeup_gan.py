import os

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.models import vgg16
from tqdm import tqdm

from network_blocks import Generator, Discriminator
from utils import num_param, histogram_matching, bbox

LIP = torch.tensor([7,9])
SKIN = torch.tensor([1,6,13])
LEYE = 4
REYE = 5

def init_weights(module):
    """ Initialize weights for convolutional layers. """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        init.xavier_normal(module.weight, gain=1.00)
        if module.bias is not None:
            init.zeros_(module.bias)   

def gan_loss(input, is_real):
    """ PatchGAN loss for GAN training. """
    loss_fn = nn.MSELoss()
    target_tensor = torch.full_like(
        input, 1.0 if is_real else 0.0, requires_grad=False,
        device = input.device, dtype=input.dtype
    )
    return loss_fn(input, target_tensor)


class MakeupGAN:
    def __init__(self, data_loader: DataLoader, args):
        self.data_loader = data_loader
        self.args = args
        self.epoch_idx = 0
        self.step_idx = 0

    
        self.construct_model()
    def save_checkpoint(self):
        """ Save model checkpoint: generator, discriminator for src and ref """
        torch.save(
            self.generator.state_dict(),
            os.path.join(
                self.args.model_dir,
                f'job_id={self.args.job_id}',
                f'G_epoch={self.epoch_idx}_step={self.step_idx}.pth'
            )
        )
        for dis_type in ['src', 'ref']:
            torch.save(
                getattr(self, f'D_{dis_type}').state_dict(),
                os.path.join(
                    self.args.model_dir,
                    f'job_id={self.args.job_id}',
                    f'D_{dis_type}_epoch={self.epoch_idx}_step={self.step_idx}.pth'
                )
            )
        print('Model checkpoint saved.'
              f'Current epoch={self.epoch_idx}, step={self.step_idx}')

    def construct_model(self):
        """ Construct generator, discriminators for makeup and non-makeup, and pretrained VGG16
            Single-GPU training only supported for ease of project implementation.. """

        assert torch.cuda.is_available(), 'CUDA is not available.'
        # Learning rate decay: (per_epoch_gamma) ** (step / total_steps_in_epoch)
        decay_lambda = lambda step: self.args.lr_gamma ** (step / len(self.data_loader))

        self.generator = Generator(
            repeat_num=self.args.gen_repeat_n,
            conv_dim=64,
            n_channel=3,
        ).cuda()
        self.generator.apply(init_weights)
        self.generator_optim = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.args.generator_lr,
            betas=(self.args.beta_1, self.args.beta_2),
        )
        self.generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.generator_optim,
            lr_lambda=decay_lambda,
        )

        self.D_src = Discriminator(
            repeat_num=self.args.dis_repeat_n,
            conv_dim=64,
            n_channel=3,
        ).cuda()
        self.D_src.apply(init_weights)
        self.D_src_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D_src.parameters()),
            lr=self.args.discriminator_lr,
            betas=(self.args.beta_1, self.args.beta_2),
        )
        self.D_src_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.D_src_optim,
            lr_lambda=decay_lambda,
        )
        self.D_ref = Discriminator(
            repeat_num=self.args.dis_repeat_n,
            conv_dim=64,
            n_channel=3,
        ).cuda()
        self.D_ref.apply(init_weights)
        self.D_ref_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D_ref.parameters()),
            lr=self.args.discriminator_lr,
            betas=(self.args.beta_1, self.args.beta_2),
        )
        self.D_ref_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.D_ref_optim,
            lr_lambda=decay_lambda,
        )
        
        self.vgg_16 = vgg16(pretrained=True).eval().cuda()

        print('construct_model(): Model constructed.')
        print('Generator size:', num_param(self.generator))
        print('Discriminator (src) size:', num_param(self.D_src))
        print('Discriminator (ref) size:', num_param(self.D_ref))

    def train(self):
        """ Training code. """
        steps_per_epoch = len(self.data_loader)
        for self.epoch_idx in tqdm(
            range(self.args.num_epochs),
            desc=f"Epoch {self.epoch_idx} out of {self.args.num_epochs}",
            total=self.args.num_epochs
        ):
            for (
                self.step_idx,
                (src_img, ref_img, src_mask, ref_mask)
                # each is tensor for batch_size * n_channel * height * width
            ) in enumerate(tqdm(
                    self.data_loader,
                    desc=f"Step {self.step_idx} out of {steps_per_epoch}",
                    total=steps_per_epoch
                )):

                src_img = src_img.cuda()
                ref_img = ref_img.cuda()

                # Loss 1. Adversarial loss (discriminator)
                dis_loss_src_real = gan_loss(self.D_src(src_img), True)
                dis_loss_ref_real = gan_loss(self.D_ref(ref_img), True)

                fake_src_img, fake_ref_img = self.generator(src_img, ref_img)
                fake_src_img = fake_src_img.detach()
                fake_ref_img = fake_ref_img.detach()
                # fake_src_img is now makeup image, so going through D_ref (makeup discriminator)
                dis_loss_ref_fake = gan_loss(self.D_ref(fake_src_img), False)
                dis_loss_src_fake = gan_loss(self.D_src(fake_ref_img), False)

                self.D_src_optim.zero_grad()
                d_src_loss = (dis_loss_src_real + dis_loss_src_fake) / 2.0
                d_src_loss.backward()
                self.D_src_optim.step()
                self.D_src_scheduler.step()
                print(f"Epoch {self.epoch_idx} step {self.step_idx}: D_src loss: {d_src_loss.item()}")
                self.D_ref_optim.zero_grad()
                d_ref_loss = (dis_loss_ref_real + dis_loss_ref_fake) / 2.0
                d_ref_loss.backward()
                self.D_ref_optim.step()
                self.D_ref_scheduler.step()
                print(f"Epoch {self.epoch_idx} step {self.step_idx}: D_ref loss: {d_ref_loss.item()}")

                # Loss 2. Identity loss
                recon_loss_fn = nn.L1Loss()
                fake_src_img_1, fake_src_img_2 = self.generator(src_img, src_img)
                fake_ref_img_1, fake_ref_img_2 = self.generator(ref_img, ref_img)
                identity_loss = (
                    recon_loss_fn(fake_src_img_1, src_img) + recon_loss_fn(fake_src_img_2, src_img)
                    + recon_loss_fn(fake_ref_img_1, ref_img) + recon_loss_fn(fake_ref_img_2, ref_img)
                ) / 4.0 * self.args.lambda_identity
                print(f"Epoch {self.epoch_idx} step {self.step_idx}: Identity loss: {identity_loss.item()}")

                # Generate fake images, will be used by loss computataion afterwards.
                fake_src_img, fake_ref_img = self.generator(src_img, ref_img)
                gen_loss_src_fake = gan_loss(self.D_src(fake_ref_img), True)
                gen_loss_ref_fake = gan_loss(self.D_ref(fake_src_img), True)

                # Loss 3. Cycle loss
                # now fake_src_img is makeup image, so going to second argument of generator
                cycle_ref_img, cycle_src_img = self.generator(fake_ref_img, fake_src_img)
                cycle_loss = (
                    recon_loss_fn(cycle_src_img, src_img)
                    + recon_loss_fn(cycle_ref_img, ref_img)
                ) / 2.0 * self.args.lambda_cycle
                print(f"Epoch {self.epoch_idx} step {self.step_idx}: Cycle loss: {cycle_loss.item()}")

                # Loss 4. Perceptual loss
                def perceptual_loss_fn(generated_img, reference_img):
                    def get_features(x):
                        for layer in self.vgg_16.features[:18]:
                            x = layer(x)
                        return x
                    loss_fn = nn.MSELoss()
                    hidden_for_generated_img = get_features(generated_img)
                    hidden_for_reference_img = get_features(reference_img).detach()
                    return loss_fn(hidden_for_generated_img, hidden_for_reference_img)
                
                perceptual_loss = (
                    perceptual_loss_fn(fake_ref_img, ref_img)
                    + perceptual_loss_fn(fake_src_img, src_img)
                ) / 2.0 * self.args.lambda_perceptual
                print(f"Epoch {self.epoch_idx} step {self.step_idx}: Perceptual loss: {perceptual_loss.item()}")

                # Loss 5. Histogram loss (Color matching loss)
                mask_src_lip = torch.isin(src_mask, LIP).float().cuda()
                mask_src_skin = torch.isin(src_mask, SKIN).float().cuda()
                mask_src_leye = bbox(torch.isin(src_mask, LEYE)).float().cuda()
                mask_src_reye = bbox(torch.isin(src_mask, REYE)).float().cuda()
                mask_ref_lip = torch.isin(ref_mask, LIP).float().cuda()
                mask_ref_skin = torch.isin(ref_mask, SKIN).float().cuda()
                mask_ref_leye = bbox(torch.isin(ref_mask, LEYE)).float().cuda()
                mask_ref_reye = bbox(torch.isin(ref_mask, REYE)).float().cuda()

                loss_lip = (
                    histogram_matching(fake_src_img, ref_img, mask_src_lip, mask_ref_lip)
                    + histogram_matching(fake_ref_img, src_img, mask_ref_lip, mask_src_lip)
                ) * self.args.lambda_lip
                loss_skin = (
                    histogram_matching(fake_src_img, ref_img, mask_src_skin, mask_ref_skin)
                    + histogram_matching(fake_ref_img, src_img, mask_ref_skin, mask_src_skin)
                ) * self.args.lambda_skin

                loss_leye = (
                    histogram_matching(fake_src_img, ref_img, mask_src_leye, mask_ref_leye)
                    + histogram_matching(fake_ref_img, src_img, mask_ref_leye, mask_src_leye)
                ) * self.args.lambda_eye

                loss_reye = (
                    histogram_matching(fake_src_img, ref_img, mask_src_reye, mask_ref_reye)
                    + histogram_matching(fake_ref_img, src_img, mask_ref_reye, mask_src_reye)
                ) * self.args.lambda_eye

                color_matching_loss = loss_lip + loss_skin + loss_leye + loss_reye
                print(f"Epoch {self.epoch_idx} step {self.step_idx}: Color matching loss: {color_matching_loss.item()}")
                
                # combine loss and backprop
                self.generator_optim.zero_grad()
                g_loss = (
                    gen_loss_src_fake + gen_loss_ref_fake
                    + identity_loss + cycle_loss + perceptual_loss
                    + color_matching_loss
                )
                g_loss.backward()
                self.generator_optim.step()
                self.generator_scheduler.step()

                # chores - saving checkpoint, visualization of training sample
                if (self.step_idx + 1) % self.args.model_interval == 0:
                    self.save_checkpoint()

                if (self.step_idx + 1) % self.args.visualization_interval == 0:
                    # Save images for visualization every visualization_interval steps
                    save_image(((
                        torch.cat([
                            src_img, ref_img,
                            fake_src_img, fake_ref_img,
                            cycle_src_img, cycle_ref_img,
                        ], dim=3) # concatenate along with width (horizontal direction)
                        + 1.0) / 2.0).clamp(0.0, 1.0), # normalize and clip for saving
                        os.path.join(
                            self.args.visualization_dir,
                            f'job_id={self.args.job_id}',
                            f'epoch={self.epoch_idx}_step={self.step_idx}.jpg'
                        ),
                        normalize=True,
                    )