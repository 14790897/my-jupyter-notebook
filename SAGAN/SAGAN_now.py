# %% [code] {"execution":{"iopub.status.busy":"2026-01-23T11:22:19.155742Z","iopub.execute_input":"2026-01-23T11:22:19.156033Z","iopub.status.idle":"2026-01-23T11:22:51.176182Z","shell.execute_reply.started":"2026-01-23T11:22:19.155999Z","shell.execute_reply":"2026-01-23T11:22:51.175213Z"},"jupyter":{"outputs_hidden":false}}
!pip install torch-fidelity
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
from torch_fidelity import calculate_metrics  # 导入FID计算模块
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.manual_seed(1)

# Self-Attention module for improved generation quality
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_attn=False):
        batch_size, C, width, height = x.size()

        # Query, Key, Value projections
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        value = self.value(x).view(batch_size, -1, width * height)

        # Attention map
        attention = F.softmax(torch.bmm(query, key), dim=-1)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Residual connection with learnable weight
        out = self.gamma * out + x
        if return_attn:
            return out, attention
        else:
            return out

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## SAGAN 参数配置 (Self-Attention GAN)

# %% [code] {"id":"xVXC_q2ekuf8","papermill":{"duration":0.079089,"end_time":"2021-10-09T06:31:24.988503","exception":false,"start_time":"2021-10-09T06:31:24.909414","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:51.178014Z","iopub.execute_input":"2026-01-23T11:22:51.178573Z","iopub.status.idle":"2026-01-23T11:22:51.250801Z","shell.execute_reply.started":"2026-01-23T11:22:51.178534Z","shell.execute_reply":"2026-01-23T11:22:51.249849Z"},"jupyter":{"outputs_hidden":false}}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16  # Increased batch size for better gradient estimation
learning_rate = 0.0002  # DCGAN recommended learning rate
num_epochs = 1500  # More epochs for small dataset
image_size = 64  # Image size (64x64)
latent_dim = 128  # Increased latent dimension for more diversity
nc = 1  # Number of channels (1 for grayscale, 3 for RGB)
ngf = 64  # Number of generator filters
ndf = 64  # Number of discriminator filters
# Training strategy parameters
d_steps = 1  # Discriminator steps per generator step
g_steps = 1  # Balanced training (1:1 ratio)
# Note: Hinge Loss doesn't use label smoothing

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## data prepare

# %% [code] {"execution":{"iopub.status.busy":"2026-01-23T11:22:51.251918Z","iopub.execute_input":"2026-01-23T11:22:51.252236Z","iopub.status.idle":"2026-01-23T11:22:52.273586Z","shell.execute_reply.started":"2026-01-23T11:22:51.252210Z","shell.execute_reply":"2026-01-23T11:22:52.272775Z"},"jupyter":{"outputs_hidden":false}}
import os
import shutil
from PIL import Image, ImageOps

# 原始目录和目标目录
source_dir = '/kaggle/input/efficientnet-data/my_label_data/0'
target_dir = './train/data'

# 创建目标目录
os.makedirs(target_dir, exist_ok=True)

shutil.copytree('/kaggle/input/efficientnet-data/efficient_net_data_me/cropped_objects/0', f'{target_dir}', dirs_exist_ok=True)
shutil.copytree('/kaggle/input/efficientnet-data/efficient2/cropped_objects/0', f'{target_dir}', dirs_exist_ok=True)

image_count = sum(1 for file_name in os.listdir(target_dir) if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')))

# 打印总数
print(f"Total images in the directory: {image_count}")

# %% [code] {"execution":{"iopub.status.busy":"2026-01-23T11:22:52.274641Z","iopub.execute_input":"2026-01-23T11:22:52.274951Z","iopub.status.idle":"2026-01-23T11:22:52.852777Z","shell.execute_reply.started":"2026-01-23T11:22:52.274910Z","shell.execute_reply":"2026-01-23T11:22:52.852031Z"},"jupyter":{"outputs_hidden":false}}
import os
import shutil
from PIL import Image
from torchvision import transforms

# --- 设置 ---
source_dir = './train/data'                  # 你的 160x160 原始图片
target_dir = './real_images_64x64_for_fid'   # 你要创建的 64x64 评估集
real_data_path = './real_images_64x64_for_fid'

# 确保目标目录存在
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir)

# 定义一个与你训练时一致的 Resize 变换
# 注意：我们只做 Resize，不做随机翻转（因为这是评估集）
# 我们使用 PIL.Image.BICUBIC 来获得最高质量的下采样
resize_transform = transforms.Compose([
    # 步骤 1: 调整大小
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
    
    # 步骤 2: 转换为灰度图
    transforms.Grayscale(num_output_channels=1)
])
print(f"正在从 {source_dir} 创建 64x64 评估集于 {target_dir}...")

# 遍历所有原始图片
count = 0
for file_name in os.listdir(source_dir):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            file_path = os.path.join(source_dir, file_name)
            img = Image.open(file_path).convert('RGB') # 确保是RGB
            
            # 应用下采样
            img_resized = resize_transform(img)
            
            # 保存到新目录
            save_path = os.path.join(target_dir, file_name)
            img_resized.save(save_path)
            count += 1
        except Exception as e:
            print(f"处理图片 {file_name} 时出错: {e}")

print(f"--- 完成！---")
print(f"总共 {count} 张 160x160 的图片被下采样并保存到了 {target_dir}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 数据加载 (Data Loading)

# %% [code] {"id":"rmKRUtX6kwt1","papermill":{"duration":12.148366,"end_time":"2021-10-09T06:31:37.214299","exception":false,"start_time":"2021-10-09T06:31:25.065933","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:52.853720Z","iopub.execute_input":"2026-01-23T11:22:52.854028Z","iopub.status.idle":"2026-01-23T11:22:52.862541Z","shell.execute_reply.started":"2026-01-23T11:22:52.854001Z","shell.execute_reply":"2026-01-23T11:22:52.861677Z"},"jupyter":{"outputs_hidden":false}}
# Optimized data transforms - minimal augmentation for better quality
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=nc),
    # Minimal augmentation - only flips for better FID
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3),
    # Removed: ColorJitter can hurt image quality metrics
    # Removed: Rotation and Affine transforms (not suitable for particle data)
    transforms.ToTensor(),
    transforms.Normalize([0.5] * nc, [0.5] * nc)  # Normalize to [-1, 1]
])

train_dataset = datasets.ImageFolder(root='./train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4
)

print(f"Dataset size: {len(train_dataset)}")
print(f"Number of batches: {len(train_loader)}")

# %% [markdown] {"papermill":{"duration":0.025809,"end_time":"2021-10-09T06:31:37.266513","exception":false,"start_time":"2021-10-09T06:31:37.240704","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# ## Data Exhibit

# %% [code] {"papermill":{"duration":0.034789,"end_time":"2021-10-09T06:31:37.327581","exception":false,"start_time":"2021-10-09T06:31:37.292792","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:52.863551Z","iopub.execute_input":"2026-01-23T11:22:52.863828Z","iopub.status.idle":"2026-01-23T11:22:53.241304Z","shell.execute_reply.started":"2026-01-23T11:22:52.863795Z","shell.execute_reply":"2026-01-23T11:22:53.240500Z"},"jupyter":{"outputs_hidden":false}}
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# 显示图像的函数
def show_images(images, nrow=8, figsize=(10, 10)):
    # 创建图像网格
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])  # 隐藏x轴刻度
    ax.set_yticks([])  # 隐藏y轴刻度
    
    # 将图片网格的大小调整并转置为 (height, width, channels)
    grid_img = make_grid(images, nrow=nrow).permute(1, 2, 0).cpu().numpy()
    
    # 显示图片
    ax.imshow(grid_img)

# 显示批次图像的函数
def show_batch(dl, n_images=64, nrow=8):
    # 只显示部分图像
    for images, _ in dl:
        # 只取前 n_images 张图像
        images = images[:n_images]
        show_images(images, nrow=nrow)
        break  # 只显示一个批次的图像

# 使用 train_loader 来展示图像
show_batch(train_loader, n_images=64, nrow=8)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Train GAN

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## FID compute method

# %% [code] {"execution":{"iopub.status.busy":"2026-01-23T11:22:53.244838Z","iopub.execute_input":"2026-01-23T11:22:53.245167Z","iopub.status.idle":"2026-01-23T11:22:53.256921Z","shell.execute_reply.started":"2026-01-23T11:22:53.245132Z","shell.execute_reply":"2026-01-23T11:22:53.255943Z"},"jupyter":{"outputs_hidden":false}}
def calculate_kid_and_fid(generator, real_data_path, device, latent_dim, 
                          num_gen_images=2000, eval_gen_batch_size=64):
    """
    在训练期间同时计算KID和FID分数 (使用 torch_fidelity)。
    
    :param generator: 当前的生成器模型。
    :param real_data_path: 真实图片所在的目录 (例如 './train/data')。
    :param device: torch.device ('cuda' 或 'cpu')。
    :param latent_dim: 潜在向量的维度。
    :param num_gen_images: 要生成多少张图片来计算KID和FID。
    :param eval_gen_batch_size: 生成图片时的批量大小。
    :return: (float, float) 计算出的KID和FID分数。
    """
    from torch_fidelity import calculate_metrics
    
    # --- 1. 创建临时文件夹保存生成的图片 ---
    gen_dir = './metrics_temp_generated'
    if os.path.exists(gen_dir):
        shutil.rmtree(gen_dir)  # 清理旧的
    os.makedirs(gen_dir)
    
    print(f"Generating {num_gen_images} images for KID and FID calculation...")
    
    # --- 2. 生成并保存图片 ---
    generator.eval()  # 切换到评估模式
    count = 0
    with torch.no_grad():
        while count < num_gen_images:
            # 确定当前批次大小
            current_batch_size = min(eval_gen_batch_size, num_gen_images - count)
            if current_batch_size <= 0:
                break
                
            noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            
            # 保存这个批次的图片
            # normalize=True: 将[-1, 1]转换为[0, 1]，然后保存为[0, 255]的PNG
            # 这是torch_fidelity期望的标准输入格式
            for i in range(current_batch_size):
                save_path = os.path.join(gen_dir, f'img_{count + i}.png')
                save_image(fake_images[i], save_path, normalize=True, value_range=(-1, 1))
            
            count += current_batch_size

    print("Generation complete. Calculating KID and FID...")

    # --- 3. 同时计算KID和FID ---
    try:
        # 计算实际图片数量用于确定subset_size
        import glob
        real_images = glob.glob(os.path.join(real_data_path, '*.[pjJbBtTgGP][npNPmMiIgGnN][gpGP]*'))
        num_real_images = len(real_images)
        
        # KID subset size: 建议为 min(num_gen_images, num_real_images) * 0.8
        kid_subset_size = int(min(num_gen_images, num_real_images) * 0.8)
        kid_subset_size = max(100, kid_subset_size)  # 至少100
        
        print(f"Using KID subset size: {kid_subset_size}")
        
        # 同时计算KID和FID
        metrics = calculate_metrics(
            input1=gen_dir,
            input2=real_data_path,
            cuda=(device.type == 'cuda'),
            kid=True,
            fid=True,
            kid_subset_size=kid_subset_size,
            verbose=False
        )
        kid_value = metrics['kernel_inception_distance_mean']
        fid_value = metrics['frechet_inception_distance']
        
        print(f"✓ KID: {kid_value:.6f}, FID: {fid_value:.4f}")
    except Exception as e:
        print(f"Metrics calculation error: {e}")
        kid_value = float('inf')
        fid_value = float('inf')
    
    # --- 4. 清理并恢复模式 ---
    shutil.rmtree(gen_dir)  # 删除临时文件夹
    generator.train()  # 恢复到训练模式
    
    return kid_value, fid_value

# %% [markdown] {"papermill":{"duration":0.037989,"end_time":"2021-10-09T06:31:39.325346","exception":false,"start_time":"2021-10-09T06:31:39.287357","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">Initializing Weights</h2>

# %% [code] {"papermill":{"duration":0.046865,"end_time":"2021-10-09T06:31:39.485146","exception":false,"start_time":"2021-10-09T06:31:39.438281","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:53.258054Z","iopub.execute_input":"2026-01-23T11:22:53.258440Z","iopub.status.idle":"2026-01-23T11:22:53.276953Z","shell.execute_reply.started":"2026-01-23T11:22:53.258382Z","shell.execute_reply":"2026-01-23T11:22:53.276176Z"},"jupyter":{"outputs_hidden":false}}
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

# %% [markdown] {"papermill":{"duration":0.037616,"end_time":"2021-10-09T06:31:39.560468","exception":false,"start_time":"2021-10-09T06:31:39.522852","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">DCGAN Generator Network</h2>

# %% [code] {"id":"_gw3SMN7jtOB","papermill":{"duration":0.051132,"end_time":"2021-10-09T06:31:39.649246","exception":false,"start_time":"2021-10-09T06:31:39.598114","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:53.277919Z","iopub.execute_input":"2026-01-23T11:22:53.278221Z","iopub.status.idle":"2026-01-23T11:22:53.288127Z","shell.execute_reply.started":"2026-01-23T11:22:53.278194Z","shell.execute_reply":"2026-01-23T11:22:53.287351Z"},"jupyter":{"outputs_hidden":false}}
class Generator(nn.Module):
    """
    SAGAN Generator (Self-Attention GAN)
    Architecture: DCGAN backbone with Self-Attention at 16x16 resolution
    Input: latent vector z of dimension (latent_dim, 1, 1)
    Output: Generated image of size (nc, 64, 64)
    Reference: Zhang et al. "Self-Attention Generative Adversarial Networks" (2018)
    """
    def __init__(self):
        super(Generator, self).__init__()
        # Input is latent_dim x 1 x 1
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        # State size: (ngf*8) x 4 x 4

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        # State size: (ngf*4) x 8 x 8

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        # State size: (ngf*2) x 16 x 16

        # Add Self-Attention at 16x16 resolution
        self.attention = SelfAttention(ngf * 2)

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        # State size: (ngf) x 32 x 32

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # State size: (nc) x 64 x 64

    def forward(self, input, return_attn=False):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 接收两个返回值
        if return_attn:
            x, attn = self.attention(x, return_attn=True)
        else:
            x = self.attention(x, return_attn=False)
        
        x = self.layer4(x)
        x = self.layer5(x)
        
        if return_attn:
            return x, attn
        else:
            return x

# %% [code] {"papermill":{"duration":3.374826,"end_time":"2021-10-09T06:31:43.061517","exception":false,"start_time":"2021-10-09T06:31:39.686691","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:53.289173Z","iopub.execute_input":"2026-01-23T11:22:53.289529Z","iopub.status.idle":"2026-01-23T11:22:53.573081Z","shell.execute_reply.started":"2026-01-23T11:22:53.289495Z","shell.execute_reply":"2026-01-23T11:22:53.572242Z"},"jupyter":{"outputs_hidden":false}}
generator = Generator().to(device)
generator.apply(weights_init)
print(generator)

# %% [code] {"papermill":{"duration":0.616425,"end_time":"2021-10-09T06:31:43.756299","exception":false,"start_time":"2021-10-09T06:31:43.139874","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:53.574093Z","iopub.execute_input":"2026-01-23T11:22:53.574508Z","iopub.status.idle":"2026-01-23T11:22:54.406367Z","shell.execute_reply.started":"2026-01-23T11:22:53.574475Z","shell.execute_reply":"2026-01-23T11:22:54.404972Z"},"jupyter":{"outputs_hidden":false}}
summary(generator, (latent_dim,1,1))

# %% [markdown] {"papermill":{"duration":0.038779,"end_time":"2021-10-09T06:31:43.835699","exception":false,"start_time":"2021-10-09T06:31:43.79692","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">DCGAN Discriminator Network</h2>

# %% [code] {"id":"xPEMXbaJCPsQ","papermill":{"duration":0.052823,"end_time":"2021-10-09T06:31:43.927067","exception":false,"start_time":"2021-10-09T06:31:43.874244","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:54.407286Z","iopub.status.idle":"2026-01-23T11:22:54.407645Z","shell.execute_reply.started":"2026-01-23T11:22:54.407492Z","shell.execute_reply":"2026-01-23T11:22:54.407511Z"},"jupyter":{"outputs_hidden":false}}
class Discriminator(nn.Module):
    """
    SAGAN Discriminator (Self-Attention GAN)
    Architecture: DCGAN backbone with Self-Attention at 16x16 + Spectral Normalization
    Input: Image of size (nc, 64, 64)
    Output: Single scalar value (probability of being real)
    Reference: Zhang et al. "Self-Attention Generative Adversarial Networks" (2018)
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        # Input is (nc) x 64 x 64
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # State size: (ndf) x 32 x 32

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # State size: (ndf*2) x 16 x 16

        # Add Self-Attention at 16x16 resolution
        self.attention = SelfAttention(ndf * 2)

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # State size: (ndf*4) x 8 x 8

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # State size: (ndf*8) x 4 x 4

        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
            # No Sigmoid for Hinge Loss (need raw logits)
        )
        # State size: 1 x 1 x 1

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.attention(x)  # Apply self-attention
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x.view(-1, 1).squeeze(1)

# %% [code] {"id":"1HJv-CnSkIuN","papermill":{"duration":0.071596,"end_time":"2021-10-09T06:31:44.037229","exception":false,"start_time":"2021-10-09T06:31:43.965633","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:54.409116Z","iopub.status.idle":"2026-01-23T11:22:54.409491Z","shell.execute_reply.started":"2026-01-23T11:22:54.409324Z","shell.execute_reply":"2026-01-23T11:22:54.409344Z"},"jupyter":{"outputs_hidden":false}}
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)

# %% [code] {"papermill":{"duration":0.0621,"end_time":"2021-10-09T06:31:44.143231","exception":false,"start_time":"2021-10-09T06:31:44.081131","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:54.410685Z","iopub.status.idle":"2026-01-23T11:22:54.411027Z","shell.execute_reply.started":"2026-01-23T11:22:54.410895Z","shell.execute_reply":"2026-01-23T11:22:54.410915Z"},"jupyter":{"outputs_hidden":false}}
summary(discriminator, (1,64,64))

# %% [code] {"id":"RFxQC7T0laZi","papermill":{"duration":0.045481,"end_time":"2021-10-09T06:31:44.228253","exception":false,"start_time":"2021-10-09T06:31:44.182772","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:54.412432Z","iopub.status.idle":"2026-01-23T11:22:54.412731Z","shell.execute_reply.started":"2026-01-23T11:22:54.412605Z","shell.execute_reply":"2026-01-23T11:22:54.412622Z"},"jupyter":{"outputs_hidden":false}}
# Hinge Loss for SAGAN (no need for nn.BCELoss)
# Discriminator Hinge Loss: E[min(0, -1 + D(x_real))] + E[min(0, -1 - D(x_fake))]
# Generator Hinge Loss: -E[D(x_fake)]

# %% [markdown] {"papermill":{"duration":0.063915,"end_time":"2021-10-09T06:31:44.635401","exception":false,"start_time":"2021-10-09T06:31:44.571486","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# ## Hinge Loss for SAGAN:
# 
# - Discriminator: maximize min(0, -1 + D(x_real)) + min(0, -1 - D(x_fake))
# - Generator: maximize D(x_fake) (or minimize -D(x_fake))

# %% [code] {"papermill":{"duration":0.072788,"end_time":"2021-10-09T06:31:44.772757","exception":false,"start_time":"2021-10-09T06:31:44.699969","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:54.413889Z","iopub.status.idle":"2026-01-23T11:22:54.414149Z","shell.execute_reply.started":"2026-01-23T11:22:54.414030Z","shell.execute_reply":"2026-01-23T11:22:54.414046Z"},"jupyter":{"outputs_hidden":false}}
# Fixed noise for visualization
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# Note: Hinge Loss doesn't need label values

# %% [code] {"id":"sis4zEVQkLf_","papermill":{"duration":0.077416,"end_time":"2021-10-09T06:31:44.914779","exception":false,"start_time":"2021-10-09T06:31:44.837363","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2026-01-23T11:22:54.416087Z","iopub.status.idle":"2026-01-23T11:22:54.416435Z","shell.execute_reply.started":"2026-01-23T11:22:54.416272Z","shell.execute_reply":"2026-01-23T11:22:54.416298Z"},"jupyter":{"outputs_hidden":false}}
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

# Learning rate schedulers for better convergence
schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=num_epochs, eta_min=1e-6)
schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=num_epochs, eta_min=1e-6)

# %% [markdown] {"papermill":{"duration":0.040677,"end_time":"2021-10-09T06:31:46.648014","exception":false,"start_time":"2021-10-09T06:31:46.607337","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">Calculate Baseline FID (Training Set Internal)</h2>

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-23T11:22:54.418535Z","iopub.status.idle":"2026-01-23T11:22:54.418960Z","shell.execute_reply.started":"2026-01-23T11:22:54.418824Z","shell.execute_reply":"2026-01-23T11:22:54.418843Z"}}
def calculate_dataset_internal_kid(data_path, test_split=0.5):
    """
    计算数据集内部的KID（将数据集分成两半并计算它们之间的KID）
    这可以作为KID评估的基线参考值
    
    :param data_path: 数据集路径
    :param test_split: 用于第二个子集的比例（默认0.5，即对半分）
    :return: 数据集内部的KID分数
    """
    from torch_fidelity import calculate_metrics
    import glob
    import random
    
    print("\n" + "=" * 50)
    print("📊 Calculating Baseline KID (Training Set Internal)")
    print("=" * 50)
    
    # 获取所有图像文件
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(data_path, ext)))
    
    total_images = len(all_images)
    print(f"Total images found: {total_images}")
    
    if total_images < 100:
        print("⚠️ Warning: Dataset too small for reliable KID calculation")
        print("   Recommended: at least 100 images")
        return None
    
    # 随机打乱并分割数据集
    random.seed(42)  # 固定随机种子以保证可重复性
    random.shuffle(all_images)
    
    split_idx = int(total_images * test_split)
    subset1_images = all_images[:split_idx]
    subset2_images = all_images[split_idx:]
    
    print(f"Subset 1: {len(subset1_images)} images")
    print(f"Subset 2: {len(subset2_images)} images")
    
    # 创建临时目录
    subset1_dir = './kid_baseline_subset1'
    subset2_dir = './kid_baseline_subset2'
    
    # 清理旧的临时目录
    if os.path.exists(subset1_dir):
        shutil.rmtree(subset1_dir)
    if os.path.exists(subset2_dir):
        shutil.rmtree(subset2_dir)
    
    os.makedirs(subset1_dir)
    os.makedirs(subset2_dir)
    
    # 复制文件到临时目录
    print("Preparing subsets...")
    for img_path in subset1_images:
        shutil.copy(img_path, os.path.join(subset1_dir, os.path.basename(img_path)))
    
    for img_path in subset2_images:
        shutil.copy(img_path, os.path.join(subset2_dir, os.path.basename(img_path)))
    
    # 计算KID
    print("Calculating KID between two subsets...")
    try:
        # 计算 KID subset size
        subset_size = int(min(len(subset1_images), len(subset2_images)) * 0.8)
        subset_size = max(100, subset_size)  # 至少100
        
        print(f"Using KID subset size: {subset_size}")
        
        metrics = calculate_metrics(
            input1=subset1_dir,
            input2=subset2_dir,
            cuda=torch.cuda.is_available(),
            kid=True,
            kid_subset_size=subset_size,
            verbose=False
        )
        baseline_kid = metrics['kernel_inception_distance_mean']
        
        print(f"\n✅ Baseline KID (Dataset Internal): {baseline_kid:.6f}")
        print(f"   This represents the 'best possible' KID for this dataset")
        print(f"   Your generator should aim to achieve KID close to or below this value")
        
    except Exception as e:
        print(f"❌ Error calculating baseline KID: {e}")
        baseline_kid = None
    
    # 清理临时目录
    shutil.rmtree(subset1_dir)
    shutil.rmtree(subset2_dir)
    
    print("=" * 50 + "\n")
    return baseline_kid

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">Training our network</h2>

# %% [code] {"execution":{"iopub.status.busy":"2026-01-23T11:22:54.420437Z","iopub.status.idle":"2026-01-23T11:22:54.420724Z","shell.execute_reply.started":"2026-01-23T11:22:54.420597Z","shell.execute_reply":"2026-01-23T11:22:54.420615Z"},"jupyter":{"outputs_hidden":false}}
import torch
import os
from torchvision.utils import save_image

# 计算训练集内部的基线KID
baseline_kid = calculate_dataset_internal_kid(real_data_path, test_split=0.5)

# Lists to keep track of progress
G_losses = []
D_losses = []
img_list = []
iters = 0

# KID and FID tracking
best_kid = float('inf')  # Initialize with infinity
best_fid = float('inf')  # Initialize with infinity
kid_scores = []  # Track KID scores over epochs
fid_scores = []  # Track FID scores over epochs

# Create directories for saving results
os.makedirs('./dcgan_weights', exist_ok=True)
os.makedirs('./dcgan_images', exist_ok=True)

# Learning rate schedulers for adaptive learning
schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=500, gamma=0.5)
schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=500, gamma=0.5)

print("Starting Optimized SAGAN (Self-Attention GAN) Training Loop...")
print(f"Architecture: DCGAN backbone + Self-Attention + Spectral Normalization")
print(f"Loss Function: Hinge Loss (recommended for SAGAN)")
print(f"Target: KID < 0.05, FID < 50")
print("-" * 50)

for epoch in range(num_epochs):
    epoch_d_loss = 0
    epoch_g_loss = 0

    for i, (real_images, _) in enumerate(train_loader):
        ############################
        # (1) Update D network with Hinge Loss
        # Hinge Loss: E[min(0, -1 + D(x_real))] + E[min(0, -1 - D(x_fake))]
        ###########################
        for _ in range(d_steps):
            discriminator.zero_grad()
            real_images_device = real_images.to(device)
            b_size = real_images_device.size(0)

            # Train with real batch using Hinge Loss
            output_real = discriminator(real_images_device)
            errD_real = torch.nn.functional.relu(1.0 - output_real).mean()
            errD_real.backward()
            D_x = output_real.mean().item()

            # Train with fake batch using Hinge Loss
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = generator(noise)
            output_fake = discriminator(fake.detach())
            errD_fake = torch.nn.functional.relu(1.0 + output_fake).mean()
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizerD.step()

        ############################
        # (2) Update G network with Hinge Loss
        # Generator Loss: -E[D(x_fake)]
        ###########################
        for _ in range(g_steps):
            generator.zero_grad()
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = generator(noise)
            # Generator wants to maximize discriminator output (minimize negative output)
            output = discriminator(fake)
            errG = -output.mean()
            errG.backward()
            D_G_z2 = output.mean().item()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        epoch_d_loss += errD.item()
        epoch_g_loss += errG.item()

        # Output training stats
        if i % 20 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f} '
                  f'LR_G: {schedulerG.get_last_lr()[0]:.6f}')

        iters += 1

    # Step the learning rate schedulers
    schedulerD.step()
    schedulerG.step()

    avg_d_loss = epoch_d_loss / len(train_loader)
    avg_g_loss = epoch_g_loss / len(train_loader)
    print(f'\nEpoch {epoch} Summary: Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}\n')

    # Check how the generator is doing by saving G's output on fixed_noise
    if (epoch % 10 == 0) or (epoch == num_epochs-1):
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        save_image(fake, f'./dcgan_images/fake_samples_epoch_{epoch:03d}.png',
                   normalize=True, value_range=(-1, 1), nrow=8)

    # Adaptive KID calculation frequency
    # More frequent early on, less frequent later
    should_calc_kid = False
    if epoch < 200:
        should_calc_kid = (epoch % 10 == 0 and epoch > 0)  # Every 10 epochs for first 200
    elif epoch < 800:
        should_calc_kid = (epoch % 20 == 0)  # Every 20 epochs from 200-800
    else:
        should_calc_kid = (epoch % 50 == 0)  # Every 50 epochs after 800
    
    if should_calc_kid or (epoch == num_epochs-1):
        print(f"\nCalculating KID and FID scores for epoch {epoch}...")
        current_kid, current_fid = calculate_kid_and_fid(
            generator=generator,
            real_data_path=real_data_path,
            device=device,
            latent_dim=latent_dim,
            num_gen_images=500,  # More samples for better metrics estimation
            eval_gen_batch_size=32
        )
        kid_scores.append((epoch, current_kid))
        fid_scores.append((epoch, current_fid))
        
        # Calculate improvement
        if len(kid_scores) > 1:
            prev_kid = kid_scores[-2][1]
            prev_fid = fid_scores[-2][1]
            kid_improvement = prev_kid - current_kid
            fid_improvement = prev_fid - current_fid
            print(f"Epoch {epoch} - KID: {current_kid:.6f} (Δ{kid_improvement:+.6f}), FID: {current_fid:.4f} (Δ{fid_improvement:+.4f})")
        else:
            print(f"Epoch {epoch} - KID: {current_kid:.6f}, FID: {current_fid:.4f}")

        # Save model if it has the best KID score so far (using KID as primary metric)
        if current_kid < best_kid:
            improvement_from_best_kid = best_kid - current_kid
            best_kid = current_kid
            best_fid = current_fid  # Also update best FID
            print(f"🎯 New best scores! KID: {best_kid:.6f} (↓{improvement_from_best_kid:.6f}), FID: {best_fid:.4f}")
            print(f"   Saving best model...")
            torch.save(generator.state_dict(), './dcgan_weights/generator_best_kid.pth')
            torch.save(discriminator.state_dict(), './dcgan_weights/discriminator_best_kid.pth')
            # Save epoch info
            with open('./dcgan_weights/best_metrics_info.txt', 'w') as f:
                f.write(f"Best KID Score: {best_kid:.6f}\n")
                f.write(f"Corresponding FID Score: {best_fid:.4f}\n")
                if baseline_kid is not None:
                    f.write(f"Baseline KID (Dataset Internal): {baseline_kid:.6f}\n")
                f.write(f"Note: Using KID as primary metric, FID as secondary\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Learning Rate: {schedulerG.get_last_lr()[0]:.6f}\n")

    # Save model checkpoints periodically
    if (epoch % 100 == 0 and epoch > 0) or (epoch == num_epochs-1):
        torch.save(generator.state_dict(), f'./dcgan_weights/generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'./dcgan_weights/discriminator_epoch_{epoch}.pth')

print("Training Complete!")
print("=" * 50)
print(f"Best Scores Achieved:")
print(f"  • KID: {best_kid:.6f}")
print(f"  • FID: {best_fid:.4f}")
print(f"Best model saved at: ./dcgan_weights/generator_best_kid.pth")

# 与基线KID对比
if baseline_kid is not None:
    print(f"\n📊 Metrics Summary:")
    print(f"   Baseline KID (Dataset Internal): {baseline_kid:.6f}")
    print(f"   Best Generated KID: {best_kid:.6f}")
    print(f"   Best Generated FID: {best_fid:.4f}")
    if best_kid < baseline_kid:
        diff = baseline_kid - best_kid
        print(f"   🎉 EXCELLENT! Generator KID is {diff:.6f} better than baseline!")
    elif best_kid < baseline_kid * 2:
        diff = best_kid - baseline_kid
        print(f"   ✓ Good! Generator KID is {diff:.6f} above baseline (within 2x)")
    else:
        diff = best_kid - baseline_kid
        print(f"   ⚠️ Generator KID is {diff:.6f} above baseline (needs improvement)")
    print(f"   Note: Using KID as primary metric, FID as secondary (both lower is better)")

if best_kid < 0.01 and best_fid < 50:
    print("\n🎉 SUCCESS! Excellent KID and FID scores achieved!")
elif best_kid < 0.05 or best_fid < 60:
    print("\n✓ Great progress! Good scores.")
else:
    print("\n⚠ Consider training longer or adjusting hyperparameters.")
print("=" * 50)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-23T11:22:54.422295Z","iopub.status.idle":"2026-01-23T11:22:54.422605Z","shell.execute_reply.started":"2026-01-23T11:22:54.422453Z","shell.execute_reply":"2026-01-23T11:22:54.422475Z"}}
# Plot the training losses and metrics
plt.figure(figsize=(20, 5))

# Subplot 1: Losses
plt.subplot(1, 3, 1)
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Subplot 2: KID Scores
if len(kid_scores) > 0:
    plt.subplot(1, 3, 2)
    epochs_kid, scores_kid = zip(*kid_scores)
    plt.title("KID Score During Training")
    plt.plot(epochs_kid, scores_kid, marker='o', color='green', linewidth=2)
    plt.axhline(y=best_kid, color='r', linestyle='--', linewidth=2, label=f'Best KID: {best_kid:.6f}')
    if baseline_kid is not None:
        plt.axhline(y=baseline_kid, color='orange', linestyle=':', linewidth=2, label=f'Baseline KID: {baseline_kid:.6f}')
    plt.xlabel("Epoch")
    plt.ylabel("KID Score")
    plt.legend()
    plt.grid(True)

# Subplot 3: FID Scores
if len(fid_scores) > 0:
    plt.subplot(1, 3, 3)
    epochs_fid, scores_fid = zip(*fid_scores)
    plt.title("FID Score During Training")
    plt.plot(epochs_fid, scores_fid, marker='s', color='blue', linewidth=2)
    plt.axhline(y=best_fid, color='r', linestyle='--', linewidth=2, label=f'Best FID: {best_fid:.4f}')
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown] {"papermill":{"duration":0.174068,"end_time":"2021-10-09T14:16:51.448795","exception":false,"start_time":"2021-10-09T14:16:51.274727","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h1 style="text-align:center;font-weight: bold">Outputing Results</h1>

# %% [code] {"_kg_hide-output":true,"papermill":{"duration":0.183417,"end_time":"2021-10-09T14:16:51.808241","exception":false,"start_time":"2021-10-09T14:16:51.624824","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-23T11:22:54.423976Z","iopub.status.idle":"2026-01-23T11:22:54.424231Z","shell.execute_reply.started":"2026-01-23T11:22:54.424108Z","shell.execute_reply":"2026-01-23T11:22:54.424124Z"}}
def getImagePaths(path):
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names[:100]  # 只取前100个

# %% [code] {"_kg_hide-output":true,"papermill":{"duration":0.417852,"end_time":"2021-10-09T14:16:52.402334","exception":false,"start_time":"2021-10-09T14:16:51.984482","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-23T11:22:54.425700Z","iopub.status.idle":"2026-01-23T11:22:54.425950Z","shell.execute_reply.started":"2026-01-23T11:22:54.425831Z","shell.execute_reply":"2026-01-23T11:22:54.425847Z"}}
import cv2
import math
import matplotlib.pyplot as plt

def display_multiple_img(images_paths):
    # 计算自适应的行列数
    num_images = len(images_paths)
    cols = int(math.ceil(math.sqrt(num_images)))  # 列数 = 根号下的图像数量，四舍五入
    rows = int(math.ceil(num_images / cols))  # 行数 = 图像数量 / 列数，四舍五入
    
    # 设置图形大小，调整到适合的比例，增加图像的显示大小
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3, rows * 3))
    
    # 遍历图像路径列表
    for ind, image_path in enumerate(images_paths):
        # 尝试读取并显示图像
        try:
            image = cv2.imread(image_path)  # 读取图像
            if image is None:
                raise ValueError(f"Image at {image_path} could not be loaded.")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            
            ax.ravel()[ind].imshow(image)  # 显示图像
            ax.ravel()[ind].set_axis_off()  # 隐藏轴
        except Exception as e:
            print(f"Error displaying image at {image_path}: {e}")
    
    # 隐藏未使用的子图（如果图像少于网格数）
    for i in range(num_images, rows * cols):
        ax.ravel()[i].set_visible(False)
    
    plt.tight_layout(pad=2.0)  # 增加子图间距
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-23T11:22:54.428201Z","iopub.status.idle":"2026-01-23T11:22:54.428661Z","shell.execute_reply.started":"2026-01-23T11:22:54.428486Z","shell.execute_reply":"2026-01-23T11:22:54.428517Z"}}
# Generate images using the best trained generator (based on KID)
generated_images_dir = './dcgan_generated'
os.makedirs(generated_images_dir, exist_ok=True)

# Load the best generator model (based on KID score)
generator_eval = Generator().to(device)
best_model_path = './dcgan_weights/generator_best_kid.pth'

# Check if best model exists, otherwise use final epoch model
if os.path.exists(best_model_path):
    print(f"Loading best model (KID: {best_kid:.6f})...")
    generator_eval.load_state_dict(torch.load(best_model_path))
else:
    print(f"Best model not found, loading final epoch model...")
    generator_eval.load_state_dict(torch.load(f'./dcgan_weights/generator_epoch_{num_epochs-1}.pth'))

generator_eval.eval()

print("Generating images...")
num_images = 600

from PIL import Image
import torchvision.transforms.functional as TF

with torch.no_grad():
    for i in range(num_images):
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        fake_image = generator_eval(noise)
        
        # 转换为 PIL 图像并使用高质量插值放大到 160x160
        # 首先反归一化：从 [-1, 1] 转到 [0, 1]
        fake_image_normalized = (fake_image + 1) / 2.0
        fake_image_normalized = torch.clamp(fake_image_normalized, 0, 1)
        
        # 转换为 PIL 图像
        pil_image = TF.to_pil_image(fake_image_normalized.squeeze(0).cpu())
        
        # 使用 LANCZOS 插值放大到 160x160（最高质量）
        # PIL.Image.LANCZOS 在新版本中被重命名为 PIL.Image.Resampling.LANCZOS
        try:
            pil_image_resized = pil_image.resize((160, 160), Image.Resampling.LANCZOS)
        except AttributeError:
            # 兼容旧版本 PIL
            pil_image_resized = pil_image.resize((160, 160), Image.LANCZOS)
            print("Warning: Using deprecated PIL.Image.LANCZOS. Consider updating Pillow library.")
        
        # 保存
        save_path = os.path.join(generated_images_dir, f'generated_{i:04d}.png')
        pil_image_resized.save(save_path)

print(f"Generated {num_images} images (160x160) in {generated_images_dir}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-23T11:22:54.429612Z","iopub.status.idle":"2026-01-23T11:22:54.429878Z","shell.execute_reply.started":"2026-01-23T11:22:54.429742Z","shell.execute_reply":"2026-01-23T11:22:54.429758Z"}}
# Create compressed archive of generated images
import zipfile
from datetime import datetime

print("\n" + "=" * 50)
print("📦 Creating compressed archive...")
print("=" * 50)

# Create archive filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
archive_name = f'generated_images_{timestamp}.zip'
archive_path = os.path.join('./', archive_name)

try:
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # Add all generated images to the archive
        image_files = [f for f in os.listdir(generated_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(generated_images_dir, img_file)
            # Add file to zip with relative path (without full directory structure)
            zipf.write(img_path, arcname=img_file)
            
            # Silent progress (only show every 100 images)
            if idx % 100 == 0 or idx == len(image_files):
                print(f"  Packed {idx}/{len(image_files)} images...", end='\r')
        
        print()  # New line after progress
    
    # Get archive size
    archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    
    print(f"\n✅ Archive created successfully!")
    print(f"   File: {archive_path}")
    print(f"   Size: {archive_size_mb:.2f} MB")
    print(f"   Images: {len(image_files)}")
    print("=" * 50 + "\n")
    
except Exception as e:
    print(f"❌ Error creating archive: {e}")
    print("=" * 50 + "\n")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-23T11:22:54.432277Z","iopub.status.idle":"2026-01-23T11:22:54.432651Z","shell.execute_reply.started":"2026-01-23T11:22:54.432455Z","shell.execute_reply":"2026-01-23T11:22:54.432485Z"}}
# Final Evaluation: Compare generated images with real data at 64x64 (same as training)
print("\n" + "=" * 50)
print("🎯 FINAL EVALUATION: Generated vs Real Data (64×64)")
print("=" * 50)
print("Using the same 64×64 comparison as training for consistent evaluation")
print("Note: This ensures fair comparison with training metrics\n")

# Use the same 64x64 real data path as training
final_real_data_path = real_data_path  # './real_images_64x64_for_fid'

# Count real images
real_count_64 = sum(1 for f in os.listdir(final_real_data_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')))

print(f"Real images (64×64): {real_count_64}")
print(f"Generated images count: {num_images}")

# Generate 64x64 images for final evaluation (without upscaling)
print("\nGenerating 64×64 images for final evaluation...")
final_eval_dir = './final_eval_generated_64x64'
if os.path.exists(final_eval_dir):
    shutil.rmtree(final_eval_dir)
os.makedirs(final_eval_dir)

generator_eval.eval()
with torch.no_grad():
    for i in range(num_images):
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        fake_image = generator_eval(noise)
        save_path = os.path.join(final_eval_dir, f'eval_{i:04d}.png')
        # Save as 64x64 (no upscaling), with proper value_range
        save_image(fake_image, save_path, normalize=True, value_range=(-1, 1))

print(f"Generated {num_images} images at 64×64 for evaluation")

# Calculate FID and KID at 64x64
print("\nCalculating FID and KID scores (64×64 vs 64×64)...")
try:
    from torch_fidelity import calculate_metrics
    
    # Calculate kid_subset_size
    kid_subset_size = int(min(num_images, real_count_64) * 0.8)
    kid_subset_size = max(100, kid_subset_size)
    
    metrics_final = calculate_metrics(
        input1=final_eval_dir,        # Generated 64x64 images
        input2=final_real_data_path,  # Real 64x64 images
        cuda=torch.cuda.is_available(),
        fid=True,
        kid=True,
        kid_subset_size=kid_subset_size,
        verbose=False
    )
    
    final_fid = metrics_final['frechet_inception_distance']
    final_kid = metrics_final['kernel_inception_distance_mean']
    
    print(f"\n✅ Final Evaluation Metrics (64×64 comparison):")
    print(f"   FID Score: {final_fid:.4f}")
    print(f"   KID Score: {final_kid:.6f}")
    
    # Compare with training metrics
    print(f"\n📊 Comparison with Training:")
    print(f"   Best During Training:")
    print(f"     - FID: {best_fid:.4f}")
    print(f"     - KID: {best_kid:.6f}")
    print(f"   Final Evaluation:")
    print(f"     - FID: {final_fid:.4f} (Δ{final_fid - best_fid:+.4f})")
    print(f"     - KID: {final_kid:.6f} (Δ{final_kid - best_kid:+.6f})")
    
    # Quality assessment
    print(f"\n🎯 Quality Assessment:")
    if final_fid < 50 and final_kid < 0.05:
        print("   🎉 EXCELLENT! Generated images match real distribution very well!")
    elif final_fid < 70 and final_kid < 0.1:
        print("   ✓ GOOD! Generated images have good quality.")
    elif final_fid < 100:
        print("   ⚠ MODERATE. Generated images show some differences from real data.")
    else:
        print("   ⚠ Generated images differ significantly from real data.")
    
    # Consistency check
    fid_diff = abs(final_fid - best_fid)
    kid_diff = abs(final_kid - best_kid)
    print(f"\n📈 Consistency Check:")
    if fid_diff < 5 and kid_diff < 0.01:
        print("   ✅ Excellent consistency with training metrics!")
    elif fid_diff < 10 and kid_diff < 0.02:
        print("   ✓ Good consistency with training metrics.")
    else:
        print("   ⚠ Some variation from training metrics (expected for different evaluation runs).")
    
    # Save final evaluation results
    with open('./dcgan_weights/final_evaluation_64x64.txt', 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("FINAL EVALUATION: 64×64 Generated vs Real Data\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Evaluation Details:\n")
        f.write(f"  - Generated Images: {num_images} (64×64, native resolution)\n")
        f.write(f"  - Real Images: {real_count_64} (64×64, from training set)\n\n")
        f.write(f"Final Evaluation Metrics (64×64 comparison):\n")
        f.write(f"  - FID Score: {final_fid:.4f}\n")
        f.write(f"  - KID Score: {final_kid:.6f}\n\n")
        f.write(f"Best Training Metrics (64×64 comparison):\n")
        f.write(f"  - Best FID: {best_fid:.4f}\n")
        f.write(f"  - Best KID: {best_kid:.6f}\n")
        if baseline_kid is not None:
            f.write(f"  - Baseline KID: {baseline_kid:.6f}\n")
        f.write(f"\nConsistency:\n")
        f.write(f"  - FID Difference: {fid_diff:.4f}\n")
        f.write(f"  - KID Difference: {kid_diff:.6f}\n")
        f.write("\n" + "=" * 50 + "\n")
    
    print(f"\n✅ Evaluation results saved to: ./dcgan_weights/final_evaluation_64x64.txt")
    
    # Clean up temporary evaluation directory
    shutil.rmtree(final_eval_dir)
    print(f"✅ Cleaned up temporary evaluation files")
    
except Exception as e:
    print(f"\n❌ Error during final evaluation: {e}")
    import traceback
    traceback.print_exc()

print("=" * 50 + "\n")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-23T11:22:54.434865Z","iopub.status.idle":"2026-01-23T11:22:54.435568Z","shell.execute_reply.started":"2026-01-23T11:22:54.435379Z","shell.execute_reply":"2026-01-23T11:22:54.435399Z"}}
display_multiple_img(getImagePaths(generated_images_dir))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-23T11:22:54.437382Z","iopub.status.idle":"2026-01-23T11:22:54.437737Z","shell.execute_reply.started":"2026-01-23T11:22:54.437611Z","shell.execute_reply":"2026-01-23T11:22:54.437628Z"}}
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

def visualize_attention(generator, device, latent_dim):
    generator.eval()
    
    # 1. 生成一张图片
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        # 调用时开启 return_attn=True
        fake_img, attn_map = generator(noise, return_attn=True)
    
    # 2. 处理图像 (Normalize back to [0,1] for display)
    img_display = (fake_img[0].cpu().detach().permute(1, 2, 0) + 1) / 2.0
    img_display = torch.clamp(img_display, 0, 1).numpy()
    
    # 3. 处理 Attention Map
    # attn_map shape: [1, N, N] 其中 N = width * height (16*16 = 256)
    # 我们需要选择一个查询点（Query Pixel），查看它关注图像的哪些区域
    
    # 设置特征图的宽高 (Self-Attention 层是在 16x16 的分辨率下做的)
    w, h = 16, 16 
    
    # 选择中心点作为查询点 (你可以修改这里选择左上角或其他位置)
    query_idx = (w // 2) * h + (h // 2)  # 中心点索引
    
    # 取出该点对所有其他像素的注意力分数
    # shape: [256] -> [16, 16]
    attention_score = attn_map[0, query_idx, :].view(w, h).cpu().detach()
    
    # 为了更好地显示，将 16x16 的热力图上采样到 64x64 (生成的图片大小)
    attention_score_upsampled = F.interpolate(
        attention_score.unsqueeze(0).unsqueeze(0), 
        size=(64, 64), 
        mode='bilinear', 
        align_corners=False
    ).squeeze().numpy()

    # 4. 绘图
    plt.figure(figsize=(12, 5))
    
    # 左图：生成的颗粒图像
    plt.subplot(1, 2, 1)
    if img_display.shape[2] == 1: # 如果是灰度图
        plt.imshow(img_display[:,:,0], cmap='gray')
    else:
        plt.imshow(img_display)
    plt.title("Generated Particle Image")
    plt.axis('off')
    
    # 右图：中心像素的 Attention Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(attention_score_upsampled, cmap='viridis', cbar=True)
    plt.title(f"Attention Map (Focus of Center Pixel)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=300)
    plt.show()
    print("Attention map generated!")

# --- 运行可视化 ---
# 确保你已经加载了最好的模型权重
# generator.load_state_dict(torch.load('./dcgan_weights/generator_best_kid.pth')) 
visualize_attention(generator, device, latent_dim)