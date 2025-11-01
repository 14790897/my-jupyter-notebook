# %% [code]
!pip install pytorch-fid
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import datetime
import matplotlib.pyplot as plt
from pytorch_fid import fid_score
import torch.nn.functional as F
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %% [markdown]
# # StyleGAN2 Implementation for Particle Detection
# 
# StyleGAN2 improvements over DCGAN/SAGAN:
# - Style-based generator with adaptive instance normalization (AdaIN)
# - Progressive growing capability
# - Path length regularization
# - No progressive growing artifacts
# - Better image quality and FID scores

# %% [markdown]
# ## StyleGAN2 Components

# %% [code]
class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate"""
    def __init__(self, in_features, out_features, bias=True, bias_init=0, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features).fill_(bias_init))
        else:
            self.bias = None
        self.scale = (1 / math.sqrt(in_features)) * lr_mul

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, bias=self.bias)
        return out


class EqualizedConv2d(nn.Module):
    """Conv2d with equalized learning rate"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size * kernel_size)

    def forward(self, x):
        out = F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)
        return out


class ModulatedConv2d(nn.Module):
    """Modulated Convolution (core of StyleGAN2)"""
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, 
                 demodulate=True, upsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.demodulate = demodulate
        
        self.scale = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.padding = kernel_size // 2
        
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        
        # Affine transformation for style modulation
        self.modulation = EqualizedLinear(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        
        # Modulation
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight.unsqueeze(0)
        weight = weight * style
        
        # Demodulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)
        
        weight = weight.view(
            batch * self.out_channels, in_channel, self.kernel_size, self.kernel_size
        )
        
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            height = height * 2  # Update height after upsampling
            width = width * 2    # Update width after upsampling
        
        x = x.view(1, batch * in_channel, height, width)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)
        
        return out


class NoiseInjection(nn.Module):
    """Inject noise for stochastic variation"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class StyledConvBlock(nn.Module):
    """StyleGAN2 Convolutional Block with style modulation"""
    def __init__(self, in_channels, out_channels, style_dim, upsample=False):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, out_channels, 3, style_dim, upsample=upsample
        )
        self.noise = NoiseInjection()
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise=None):
        out = self.conv(x, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class MappingNetwork(nn.Module):
    """Mapping network Z -> W"""
    def __init__(self, latent_dim=512, style_dim=512, n_layers=8):
        super().__init__()
        
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(
                latent_dim if i == 0 else style_dim, 
                style_dim
            ))
            layers.append(nn.LeakyReLU(0.2))
        
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


class ToRGB(nn.Module):
    """Convert features to RGB image"""
    def __init__(self, in_channels, style_dim, out_channels=1):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias
        
        if skip is not None:
            out = out + skip
        
        return out


class StyleGAN2Generator(nn.Module):
    """
    StyleGAN2 Generator (Optimized for small datasets)
    Reference: Karras et al. "Analyzing and Improving the Image Quality of StyleGAN" (2020)

    Modifications for small dataset (300 images):
    - Reduced style_dim: 512 -> 256
    - Reduced mapping network layers: 8 -> 4
    - Reduced channel counts throughout
    """
    def __init__(self, latent_dim=128, style_dim=256, n_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        # Mapping network - REDUCED layers for small dataset
        self.mapping = MappingNetwork(latent_dim, style_dim, n_layers=4)

        # Constant input - REDUCED channels
        self.constant = nn.Parameter(torch.randn(1, 256, 4, 4))

        # Synthesis network - REDUCED channel counts
        # 4x4
        self.conv1 = StyledConvBlock(256, 256, style_dim)
        self.to_rgb1 = ToRGB(256, style_dim, n_channels)

        # 4->8
        self.conv2 = StyledConvBlock(256, 256, style_dim, upsample=True)
        self.conv3 = StyledConvBlock(256, 256, style_dim)
        self.to_rgb2 = ToRGB(256, style_dim, n_channels)

        # 8->16
        self.conv4 = StyledConvBlock(256, 128, style_dim, upsample=True)
        self.conv5 = StyledConvBlock(128, 128, style_dim)
        self.to_rgb3 = ToRGB(128, style_dim, n_channels)

        # 16->32
        self.conv6 = StyledConvBlock(128, 64, style_dim, upsample=True)
        self.conv7 = StyledConvBlock(64, 64, style_dim)
        self.to_rgb4 = ToRGB(64, style_dim, n_channels)

        # 32->64
        self.conv8 = StyledConvBlock(64, 32, style_dim, upsample=True)
        self.conv9 = StyledConvBlock(32, 32, style_dim)
        self.to_rgb5 = ToRGB(32, style_dim, n_channels)

    def forward(self, z, return_latents=False):
        # Map to W space
        w = self.mapping(z)
        
        # For path length regularization, we need w to have gradients
        if return_latents:
            w_for_grad = w.clone().requires_grad_(True)
        else:
            w_for_grad = w
        
        # Start from constant
        batch = z.shape[0]
        x = self.constant.repeat(batch, 1, 1, 1)
        
        # Synthesis network with progressive RGB outputs
        x = self.conv1(x, w_for_grad)
        
        x = self.conv2(x, w_for_grad)
        x = self.conv3(x, w_for_grad)
        
        x = self.conv4(x, w_for_grad)
        x = self.conv5(x, w_for_grad)
        
        x = self.conv6(x, w_for_grad)
        x = self.conv7(x, w_for_grad)
        
        x = self.conv8(x, w_for_grad)
        x = self.conv9(x, w_for_grad)
        rgb = self.to_rgb5(x, w_for_grad)
        
        rgb = torch.tanh(rgb)  # Output in [-1, 1]
        
        if return_latents:
            return rgb, w_for_grad
        return rgb


class StyleGAN2Discriminator(nn.Module):
    """
    StyleGAN2 Discriminator (Optimized for small datasets)

    Modifications for small dataset (300 images):
    - Reduced channel counts to prevent overfitting
    - Maintains residual structure
    """
    def __init__(self, n_channels=1):
        super().__init__()

        # From RGB - REDUCED initial channels
        self.from_rgb = EqualizedConv2d(n_channels, 32, 1)

        # Residual blocks with REDUCED channel counts
        # 64x64 -> 32x32
        self.conv1 = EqualizedConv2d(32, 64, 3, padding=1)
        self.conv2 = EqualizedConv2d(64, 64, 3, padding=1)

        # 32x32 -> 16x16
        self.conv3 = EqualizedConv2d(64, 128, 3, padding=1)
        self.conv4 = EqualizedConv2d(128, 128, 3, padding=1)

        # 16x16 -> 8x8
        self.conv5 = EqualizedConv2d(128, 256, 3, padding=1)
        self.conv6 = EqualizedConv2d(256, 256, 3, padding=1)

        # 8x8 -> 4x4
        self.conv7 = EqualizedConv2d(256, 256, 3, padding=1)
        self.conv8 = EqualizedConv2d(256, 256, 3, padding=1)

        # Final layers
        self.final_conv = EqualizedConv2d(256, 256, 4)
        self.final_linear = EqualizedLinear(256, 1)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.activation(x)
        
        # 64x64 -> 32x32
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, 2)
        
        # 32x32 -> 16x16
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, 2)
        
        # 16x16 -> 8x8
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, 2)
        
        # 8x8 -> 4x4
        x = self.conv7(x)
        x = self.activation(x)
        x = self.conv8(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, 2)
        
        # 4x4 -> 1x1
        x = self.final_conv(x)
        x = self.activation(x)
        
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)
        
        return x


# %% [markdown]
# ## StyleGAN2 参数配置

# %% [code]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Optimized parameters for small dataset (300 images)
batch_size = 16  # Increased for better gradient estimation
learning_rate_g = 0.001  # Reduced to prevent overfitting
learning_rate_d = 0.001  # Reduced to prevent overfitting
num_epochs = 3000  # More epochs for small dataset
image_size = 64
latent_dim = 128  # Z space dimension
style_dim = 256   # W space dimension - REDUCED from 512 for small dataset
nc = 1  # Grayscale

# StyleGAN2 specific parameters - tuned for small dataset
r1_gamma = 5.0  # R1 regularization weight - reduced
path_length_penalty = 1.0  # Path length regularization weight - reduced
lazy_regularization = 8  # Apply regularization more frequently
ema_decay = 0.999  # EMA decay rate for generator

# %% [markdown]
# ## 数据准备和加载

# %% [code]
import os
import shutil
from PIL import Image, ImageOps

# 原始目录和目标目录
source_dir = '/kaggle/input/efficientnet-data/my_label_data/0'
target_dir = './train/data'

# 创建目标目录
os.makedirs(target_dir, exist_ok=True)

shutil.copytree('/kaggle/input/efficientnet-data/efficient_net_data_me/cropped_objects/0', 
                f'{target_dir}', dirs_exist_ok=True)
shutil.copytree('/kaggle/input/efficientnet-data/efficient2/cropped_objects/0', 
                f'{target_dir}', dirs_exist_ok=True)

image_count = sum(1 for file_name in os.listdir(target_dir) 
                  if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')))

print(f"Total images in the directory: {image_count}")

# %% [code]
# 创建 64x64 评估集
source_dir = './train/data'
target_dir = './real_images_64x64_for_fid'
real_data_path = './real_images_64x64_for_fid'

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir)

resize_transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=1)
])

print(f"Creating 64x64 evaluation set from {source_dir}...")

count = 0
for file_name in os.listdir(source_dir):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            file_path = os.path.join(source_dir, file_name)
            img = Image.open(file_path).convert('RGB')
            img_resized = resize_transform(img)
            save_path = os.path.join(target_dir, file_name)
            img_resized.save(save_path)
            count += 1
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print(f"Completed! {count} images saved to {target_dir}")

# %% [markdown]
# ## 数据加载

# %% [code]
# Enhanced data augmentation for small dataset
# Conservative augmentation suitable for particle detection
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=nc),
    # Only use augmentations that preserve particle physics properties
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # Brightness/contrast changes to simulate different imaging conditions
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * nc, [0.5] * nc),
    # Random erasing simulates occlusion/noise
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
])

train_dataset = datasets.ImageFolder(root='./train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    drop_last=True  # Important for StyleGAN2
)

print(f"Dataset size: {len(train_dataset)}")
print(f"Number of batches: {len(train_loader)}")

# %% [markdown]
# ## Data Visualization

# %% [code]
def show_images(images, nrow=8, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    grid_img = make_grid(images, nrow=nrow).permute(1, 2, 0).cpu().numpy()
    ax.imshow(grid_img)

def show_batch(dl, n_images=64, nrow=8):
    for images, _ in dl:
        images = images[:n_images]
        show_images(images, nrow=nrow)
        break

show_batch(train_loader, n_images=min(64, batch_size * 8), nrow=8)

# %% [markdown]
# ## Differentiable Augmentation (DiffAugment)

# %% [code]
def DiffAugment(x, policy='color,translation,cutout'):
    """
    Differentiable Augmentation for Data-Efficient GAN Training
    Reference: Zhao et al. "Training Generative Adversarial Networks with Limited Data" (NeurIPS 2020)

    This is critical for small datasets (like 300 images) to prevent overfitting.
    """
    if policy:
        if 'color' in policy:
            # Color jittering
            x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
            x = torch.clamp(x, -1, 1)

        if 'translation' in policy:
            # Random translation (12.5% of image size)
            shift_x, shift_y = int(x.size(2) * 0.125), int(x.size(3) * 0.125)
            translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
            translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)

            grid_batch, grid_x, grid_y = torch.meshgrid(
                torch.arange(x.size(0), dtype=torch.long, device=x.device),
                torch.arange(x.size(2), dtype=torch.long, device=x.device),
                torch.arange(x.size(3), dtype=torch.long, device=x.device),
                indexing='ij'
            )
            grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
            grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
            x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
            x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()

        if 'cutout' in policy:
            # Random cutout (50% of image size)
            cutout_size = int(x.size(2) * 0.5)
            offset_x = torch.randint(0, x.size(2) + (1 - cutout_size % 2), size=[x.size(0), 1, 1], device=x.device)
            offset_y = torch.randint(0, x.size(3) + (1 - cutout_size % 2), size=[x.size(0), 1, 1], device=x.device)

            grid_batch, grid_x, grid_y = torch.meshgrid(
                torch.arange(x.size(0), dtype=torch.long, device=x.device),
                torch.arange(cutout_size, dtype=torch.long, device=x.device),
                torch.arange(cutout_size, dtype=torch.long, device=x.device),
                indexing='ij'
            )
            grid_x = torch.clamp(grid_x + offset_x - cutout_size // 2, min=0, max=x.size(2) - 1)
            grid_y = torch.clamp(grid_y + offset_y - cutout_size // 2, min=0, max=x.size(3) - 1)
            mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
            mask[grid_batch, grid_x, grid_y] = 0
            x = x * mask.unsqueeze(1)

    return x


# %% [markdown]
# ## FID Calculation

# %% [code]
def calculate_fid(generator, real_data_path, device, latent_dim, 
                  num_gen_images=500, eval_gen_batch_size=32, 
                  fid_calc_batch_size=50, dims=2048):
    """Calculate FID score during training"""
    gen_dir = './fid_temp_generated'
    if os.path.exists(gen_dir):
        shutil.rmtree(gen_dir)
    os.makedirs(gen_dir)
    
    print(f"Generating {num_gen_images} images for FID calculation...")
    
    generator.eval()
    count = 0
    with torch.no_grad():
        while count < num_gen_images:
            current_batch_size = min(eval_gen_batch_size, num_gen_images - count)
            if current_batch_size <= 0:
                break
                
            noise = torch.randn(current_batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            
            for i in range(current_batch_size):
                save_path = os.path.join(gen_dir, f'img_{count + i}.png')
                save_image(fake_images[i], save_path, normalize=True)
            
            count += current_batch_size

    print("Generation complete. Calculating FID...")

    paths = [real_data_path, gen_dir]
    fid_value = fid_score.calculate_fid_given_paths(
        paths=paths,
        batch_size=fid_calc_batch_size,
        device=device,
        dims=dims
    )
    
    shutil.rmtree(gen_dir)
    generator.train()
    
    return fid_value


# %% [markdown]
# ## Weight Initialization

# %% [code]
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# %% [markdown]
# ## Initialize Models

# %% [code]
generator = StyleGAN2Generator(latent_dim=latent_dim, style_dim=style_dim, n_channels=nc).to(device)
discriminator = StyleGAN2Discriminator(n_channels=nc).to(device)

print("=" * 60)
print("StyleGAN2 Generator")
print("=" * 60)
print(generator)
print("\n" + "=" * 60)
print("StyleGAN2 Discriminator")
print("=" * 60)
print(discriminator)

# %% [markdown]
# ## Helper Functions

# %% [code]
def requires_grad(model, flag=True):
    """Set requires_grad for all parameters"""
    for p in model.parameters():
        p.requires_grad = flag


# %% [markdown]
# ## Loss Functions and Regularization

# %% [code]
def d_logistic_loss(real_pred, fake_pred):
    """Non-saturating logistic loss for discriminator"""
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    """Non-saturating loss for generator"""
    return F.softplus(-fake_pred).mean()


def d_r1_loss(real_pred, real_img):
    """R1 regularization for discriminator"""
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    """Path length regularization for generator"""
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    # latents is (batch, style_dim), so we sum over style_dim (dim=1)
    path_lengths = torch.sqrt(grad.pow(2).sum(1))
    
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    
    path_penalty = (path_lengths - path_mean).pow(2).mean()
    
    return path_penalty, path_mean.detach()


# %% [markdown]
# ## Optimizers

# %% [code]
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.0, 0.99))
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate_g, betas=(0.0, 0.99))

# Learning rate schedulers
schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=num_epochs, eta_min=1e-6)
schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=num_epochs, eta_min=1e-6)

# %% [markdown]
# ## Training Loop

# %% [code]
# Training state
G_losses = []
D_losses = []
fid_scores = []
best_fid = float('inf')
mean_path_length = 0

# Create directories
os.makedirs('./stylegan2_weights', exist_ok=True)
os.makedirs('./stylegan2_images', exist_ok=True)

# Fixed noise for visualization
fixed_noise = torch.randn(64, latent_dim, device=device)

# ⭐ Create EMA generator for better image quality
from copy import deepcopy
g_ema = deepcopy(generator).eval()
requires_grad(g_ema, False)

print("=" * 60)
print("Starting StyleGAN2 Training (Optimized for Small Dataset)")
print("=" * 60)
print(f"Target: FID < 50")
print(f"Dataset size: ~300 images")
print(f"Batch size: {batch_size}")
print(f"Learning rate G: {learning_rate_g}, D: {learning_rate_d}")
print(f"Using DiffAugment: YES (critical for small datasets)")
print(f"Using EMA: YES (decay={ema_decay})")
print("=" * 60)

for epoch in range(num_epochs):
    epoch_d_loss = 0
    epoch_g_loss = 0
    
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        b_size = real_images.size(0)

        # ==========================================
        # Train Discriminator
        # ==========================================
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # Generate fake images
        noise = torch.randn(b_size, latent_dim, device=device)
        fake_images = generator(noise)

        # ⭐ Apply DiffAugment to BOTH real and fake images
        real_images_aug = DiffAugment(real_images, policy='color,translation,cutout')
        fake_images_aug = DiffAugment(fake_images, policy='color,translation,cutout')

        # Discriminator predictions on augmented images
        real_pred = discriminator(real_images_aug)
        fake_pred = discriminator(fake_images_aug.detach())

        # Discriminator loss
        d_loss = d_logistic_loss(real_pred, fake_pred)

        discriminator.zero_grad()
        d_loss.backward()
        optimizerD.step()

        # R1 regularization (lazy) - apply to original real images
        if i % lazy_regularization == 0:
            real_images.requires_grad = True
            real_pred = discriminator(real_images)
            r1_loss = d_r1_loss(real_pred, real_images)

            discriminator.zero_grad()
            (r1_gamma / 2 * r1_loss * lazy_regularization).backward()
            optimizerD.step()

        # ==========================================
        # Train Generator
        # ==========================================
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = torch.randn(b_size, latent_dim, device=device)
        fake_images, latents = generator(noise, return_latents=True)

        # ⭐ Apply DiffAugment to fake images
        fake_images_aug = DiffAugment(fake_images, policy='color,translation,cutout')

        fake_pred = discriminator(fake_images_aug)
        g_loss = g_nonsaturating_loss(fake_pred)

        generator.zero_grad()
        g_loss.backward()
        optimizerG.step()

        # ⭐ Update EMA generator
        with torch.no_grad():
            for p_ema, p in zip(g_ema.parameters(), generator.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_decay))

        # Path length regularization (lazy)
        if i % lazy_regularization == 0:
            noise = torch.randn(b_size, latent_dim, device=device)
            fake_images, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length = g_path_regularize(
                fake_images, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = path_length_penalty * path_loss
            if weighted_path_loss.item() > 0:
                (weighted_path_loss * lazy_regularization).backward()
            optimizerG.step()

            # ⭐ Update EMA after path regularization too
            with torch.no_grad():
                for p_ema, p in zip(g_ema.parameters(), generator.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_decay))
        
        # Track losses
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        
        # Print progress
        if i % 20 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] '
                  f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f} '
                  f'LR_G: {schedulerG.get_last_lr()[0]:.6f}')
    
    # Step schedulers
    schedulerD.step()
    schedulerG.step()
    
    avg_d_loss = epoch_d_loss / len(train_loader)
    avg_g_loss = epoch_g_loss / len(train_loader)
    print(f'\nEpoch {epoch} Summary: Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}\n')
    
    # Save sample images - use EMA generator for best quality
    if (epoch % 10 == 0) or (epoch == num_epochs-1):
        with torch.no_grad():
            fake = g_ema(fixed_noise).detach().cpu()
        save_image(fake, f'./stylegan2_images/fake_samples_epoch_{epoch:03d}.png',
                   normalize=True, nrow=8)

    # Calculate FID - use EMA generator
    should_calc_fid = False
    if epoch < 200:
        should_calc_fid = (epoch % 10 == 0 and epoch > 0)
    elif epoch < 800:
        should_calc_fid = (epoch % 20 == 0)
    else:
        should_calc_fid = (epoch % 50 == 0)

    if should_calc_fid or (epoch == num_epochs-1):
        print(f"\nCalculating FID score for epoch {epoch}...")
        current_fid = calculate_fid(
            generator=g_ema,  # ⭐ Use EMA generator for FID
            real_data_path=real_data_path,
            device=device,
            latent_dim=latent_dim,
            num_gen_images=500,
            eval_gen_batch_size=32,
            fid_calc_batch_size=50,
            dims=2048
        )
        fid_scores.append((epoch, current_fid))

        if len(fid_scores) > 1:
            prev_fid = fid_scores[-2][1]
            improvement = prev_fid - current_fid
            print(f"Epoch {epoch} - FID Score: {current_fid:.4f} (Change: {improvement:+.4f})")
        else:
            print(f"Epoch {epoch} - FID Score: {current_fid:.4f}")

        # Save best model - save both regular and EMA
        if current_fid < best_fid:
            improvement_from_best = best_fid - current_fid
            best_fid = current_fid
            print(f"🎯 New best FID score: {best_fid:.4f} (Improved by {improvement_from_best:.4f})")
            print(f"   Saving best model...")
            torch.save(generator.state_dict(), './stylegan2_weights/generator_best_fid.pth')
            torch.save(g_ema.state_dict(), './stylegan2_weights/generator_ema_best_fid.pth')  # ⭐ Save EMA
            torch.save(discriminator.state_dict(), './stylegan2_weights/discriminator_best_fid.pth')
            with open('./stylegan2_weights/best_fid_info.txt', 'w') as f:
                f.write(f"Best FID Score: {best_fid:.4f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Learning Rate: {schedulerG.get_last_lr()[0]:.6f}\n")
                f.write(f"Using EMA: True\n")
                f.write(f"Using DiffAugment: True\n")

    # Save checkpoints
    if (epoch % 100 == 0 and epoch > 0) or (epoch == num_epochs-1):
        torch.save(generator.state_dict(), f'./stylegan2_weights/generator_epoch_{epoch}.pth')
        torch.save(g_ema.state_dict(), f'./stylegan2_weights/generator_ema_epoch_{epoch}.pth')  # ⭐ Save EMA
        torch.save(discriminator.state_dict(), f'./stylegan2_weights/discriminator_epoch_{epoch}.pth')

print("=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Best FID Score Achieved: {best_fid:.4f}")
print(f"Best model saved at: ./stylegan2_weights/generator_best_fid.pth")
if best_fid < 50:
    print("🎉 SUCCESS! FID target (<50) achieved!")
elif best_fid < 60:
    print("✓ Great progress! Close to target.")
else:
    print("⚠ Consider training longer or adjusting hyperparameters.")
print("=" * 60)

# %% [markdown]
# ## Visualization

# %% [code]
# Plot losses and FID
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G", alpha=0.7)
plt.plot(D_losses, label="D", alpha=0.7)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

if len(fid_scores) > 0:
    plt.subplot(1, 2, 2)
    epochs_fid, scores_fid = zip(*fid_scores)
    plt.title("FID Score During Training")
    plt.plot(epochs_fid, scores_fid, marker='o', color='green')
    plt.axhline(y=best_fid, color='r', linestyle='--', label=f'Best FID: {best_fid:.4f}')
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Generate Final Images

# %% [code]
import cv2

def getImagePaths(path):
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names[:100]

def display_multiple_img(images_paths):
    num_images = len(images_paths)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3, rows * 3))
    
    for ind, image_path in enumerate(images_paths):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image at {image_path} could not be loaded.")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.ravel()[ind].imshow(image)
            ax.ravel()[ind].set_axis_off()
        except Exception as e:
            print(f"Error displaying image at {image_path}: {e}")
    
    for i in range(num_images, rows * cols):
        ax.ravel()[i].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    plt.show()

# Generate images using best model
generated_images_dir = './stylegan2_generated'
os.makedirs(generated_images_dir, exist_ok=True)

generator_eval = StyleGAN2Generator(latent_dim=latent_dim, style_dim=style_dim, n_channels=nc).to(device)
best_model_path = './stylegan2_weights/generator_ema_best_fid.pth'  # ⭐ Use EMA model

if os.path.exists(best_model_path):
    print(f"Loading best EMA model (FID: {best_fid:.4f})...")
    generator_eval.load_state_dict(torch.load(best_model_path))
else:
    print(f"EMA model not found, trying regular best model...")
    fallback_path = './stylegan2_weights/generator_best_fid.pth'
    if os.path.exists(fallback_path):
        generator_eval.load_state_dict(torch.load(fallback_path))
    else:
        print(f"No best model found, loading final epoch model...")
        generator_eval.load_state_dict(torch.load(f'./stylegan2_weights/generator_ema_epoch_{num_epochs-1}.pth'))

generator_eval.eval()

print("Generating images with StyleGAN2 (EMA)...")
num_images = 100
with torch.no_grad():
    for i in range(num_images):
        noise = torch.randn(1, latent_dim, device=device)
        fake_image = generator_eval(noise)
        save_path = os.path.join(generated_images_dir, f'generated_{i:04d}.png')
        save_image(fake_image, save_path, normalize=True)

print(f"Generated {num_images} images in {generated_images_dir}")

# Display generated images
display_multiple_img(getImagePaths(generated_images_dir))

# %% [markdown]
# ## Summary
# 
# This script implements StyleGAN2 with the following key features:
# 
# 1. **Style-based Generator**: Uses adaptive instance normalization (AdaIN) for style control
# 2. **Modulated Convolution**: Core building block replacing traditional convolutions
# 3. **Mapping Network**: Maps latent code Z to intermediate latent space W
# 4. **Path Length Regularization**: Encourages smooth mapping from W to images
# 5. **R1 Regularization**: Gradient penalty for discriminator
# 6. **Non-saturating Loss**: Improved training stability
# 
# StyleGAN2 typically achieves better FID scores than DCGAN/SAGAN due to:
# - Better disentanglement of image features
# - More stable training
# - Higher quality image generation
# - Better control over generated images
