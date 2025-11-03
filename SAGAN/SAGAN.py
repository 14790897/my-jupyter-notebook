# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:45:36.175246Z","iopub.execute_input":"2025-10-29T08:45:36.175473Z","iopub.status.idle":"2025-10-29T08:47:17.862566Z","shell.execute_reply.started":"2025-10-29T08:45:36.175428Z","shell.execute_reply":"2025-10-29T08:47:17.861917Z"},"jupyter":{"outputs_hidden":false}}
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
from torch_fidelity import calculate_metrics  # 导入FID和KID计算模块
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

    def forward(self, x):
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
        return out

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## SAGAN 参数配置 (Self-Attention GAN)

# %% [code] {"id":"xVXC_q2ekuf8","papermill":{"duration":0.079089,"end_time":"2021-10-09T06:31:24.988503","exception":false,"start_time":"2021-10-09T06:31:24.909414","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:17.864024Z","iopub.execute_input":"2025-10-29T08:47:17.864499Z","iopub.status.idle":"2025-10-29T08:47:17.926536Z","shell.execute_reply.started":"2025-10-29T08:47:17.864478Z","shell.execute_reply":"2025-10-29T08:47:17.925750Z"},"jupyter":{"outputs_hidden":false}}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16  # Increased for better gradient estimation and stability
learning_rate = 0.0002  # Lower learning rate for more stable training
num_epochs = 2000  # More epochs for better convergence
image_size = 64  # Image size (64x64)
latent_dim = 128  # Increased latent dimension for more diversity
nc = 1  # Number of channels (1 for grayscale, 3 for RGB)
ngf = 64  # Generator filters (64→96, +50% capacity for better feature learning)
ndf = 64  # Discriminator filters (64→96, +50% capacity for better discrimination)
# Training strategy parameters
d_steps = 1  # Discriminator steps per generator step
g_steps = 1  # More generator steps to prevent discriminator dominance

# ============ LOSS FUNCTION CONFIGURATION ============
# Choose loss function: 'bce' or 'hinge'
loss_type = 'hinge'  # Options: 'bce' (BCE Loss) or 'hinge' (Hinge Loss)
# =====================================================

# Label smoothing for better training stability (only used with BCE)
real_label_smooth = 0.95  # Real labels = 0.95 instead of 1.0 (reduced smoothing)
fake_label_smooth = 0.0  # Fake labels = 0.0

# Progressive training parameters
use_progressive_gan = True  # Enable progressive training strategy
warmup_epochs = 100  # Epochs for warmup phase

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## data prepare

# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:47:17.927412Z","iopub.execute_input":"2025-10-29T08:47:17.927751Z","iopub.status.idle":"2025-10-29T08:47:19.348568Z","shell.execute_reply.started":"2025-10-29T08:47:17.927723Z","shell.execute_reply":"2025-10-29T08:47:19.347926Z"},"jupyter":{"outputs_hidden":false}}
import os
import shutil
from PIL import Image, ImageOps

# 原始目录和目标目录
# source_dir = '/kaggle/input/efficientnet-data/my_label_data/0'
target_dir = './train/data'

# 创建目标目录
os.makedirs(target_dir, exist_ok=True)

shutil.copytree('/kaggle/input/efficientnet-data/efficient_net_data_me/cropped_objects/0', f'{target_dir}', dirs_exist_ok=True)
shutil.copytree('/kaggle/input/efficientnet-data/efficient2/cropped_objects/0', f'{target_dir}', dirs_exist_ok=True)

image_count = sum(1 for file_name in os.listdir(target_dir) if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')))

# 打印总数
print(f"Total images in the directory: {image_count}")

# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:47:19.349317Z","iopub.execute_input":"2025-10-29T08:47:19.349601Z","iopub.status.idle":"2025-10-29T08:47:19.951009Z","shell.execute_reply.started":"2025-10-29T08:47:19.349571Z","shell.execute_reply":"2025-10-29T08:47:19.950319Z"},"jupyter":{"outputs_hidden":false}}
import os
import shutil
from PIL import Image
from torchvision import transforms

# --- 设置 ---
source_dir = './train/data'                  # 你的 160x160 原始图片
target_dir = './real_images_64x64_for_fid'   # 你要创建的 64x64 评估集
real_data_path = './real_images_64x64_for_fid'

# 测试集路径 (用于最终FID评估，不参与训练)
test_source_dir = '/kaggle/input/efficientnet-data/test/0'
test_target_dir = './test_images_64x64_for_fid'
test_data_path = './test_images_64x64_for_fid'

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

# --- 准备测试集 (用于最终FID评估) ---
if os.path.exists(test_target_dir):
    shutil.rmtree(test_target_dir)
os.makedirs(test_target_dir)

print(f"\n正在从 {test_source_dir} 创建 64x64 测试集于 {test_target_dir}...")

# 遍历测试集图片
test_count = 0
for file_name in os.listdir(test_source_dir):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            file_path = os.path.join(test_source_dir, file_name)
            img = Image.open(file_path).convert('RGB')
            
            # 应用下采样
            img_resized = resize_transform(img)
            
            # 保存到新目录
            save_path = os.path.join(test_target_dir, file_name)
            img_resized.save(save_path)
            test_count += 1
        except Exception as e:
            print(f"处理测试图片 {file_name} 时出错: {e}")

print(f"--- 测试集准备完成！---")
print(f"总共 {test_count} 张测试集图片被下采样并保存到了 {test_target_dir}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 数据加载 (Data Loading)

# %% [code] {"id":"rmKRUtX6kwt1","papermill":{"duration":12.148366,"end_time":"2021-10-09T06:31:37.214299","exception":false,"start_time":"2021-10-09T06:31:25.065933","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:19.951886Z","iopub.execute_input":"2025-10-29T08:47:19.952156Z","iopub.status.idle":"2025-10-29T08:47:19.960161Z","shell.execute_reply.started":"2025-10-29T08:47:19.952132Z","shell.execute_reply":"2025-10-29T08:47:19.959481Z"},"jupyter":{"outputs_hidden":false}}
# ============ TRAINING DATA AUGMENTATION ============
# Training: Use augmentation to increase data diversity
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=nc),
    # Data augmentation for training only
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # Removed ColorJitter - can hurt FID and may not match real data distribution
    transforms.ToTensor(),
    transforms.Normalize([0.5] * nc, [0.5] * nc)  # Normalize to [-1, 1]
])

# ============ IMPORTANT NOTE ============
# The real images for FID calculation (./real_images_64x64_for_fid and 
# ./test_images_64x64_for_fid) were prepared WITHOUT any augmentation.
# This is correct because:
# 1. FID measures distribution distance between real and generated images
# 2. Real distribution should reflect the original data, not augmented data
# 3. Augmentation would artificially expand the real distribution
# =======================================

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

# %% [code] {"papermill":{"duration":0.034789,"end_time":"2021-10-09T06:31:37.327581","exception":false,"start_time":"2021-10-09T06:31:37.292792","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:19.960861Z","iopub.execute_input":"2025-10-29T08:47:19.961051Z","iopub.status.idle":"2025-10-29T08:47:20.786344Z","shell.execute_reply.started":"2025-10-29T08:47:19.961035Z","shell.execute_reply":"2025-10-29T08:47:20.785425Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:47:20.789251Z","iopub.execute_input":"2025-10-29T08:47:20.789532Z","iopub.status.idle":"2025-10-29T08:47:20.797502Z","shell.execute_reply.started":"2025-10-29T08:47:20.789510Z","shell.execute_reply":"2025-10-29T08:47:20.796777Z"},"jupyter":{"outputs_hidden":false}}
def calculate_fid(generator, real_data_path, device, latent_dim, 
                  num_gen_images=2000, eval_gen_batch_size=64):
    """
    在训练期间计算FID分数 (使用 torch_fidelity)。
    
    :param generator: 当前的生成器模型。
    :param real_data_path: 真实图片所在的目录 (例如 './train/data')。
    :param device: torch.device ('cuda' 或 'cpu')。
    :param latent_dim: 潜在向量的维度。
    :param num_gen_images: 要生成多少张图片来计算FID。
    :param eval_gen_batch_size: 生成图片时的批量大小。
    :return: (float) 计算出的FID分数。
    """
    from torch_fidelity import calculate_metrics
    
    # --- 1. 创建临时文件夹保存生成的图片 ---
    gen_dir = './fid_temp_generated'
    if os.path.exists(gen_dir):
        shutil.rmtree(gen_dir)  # 清理旧的
    os.makedirs(gen_dir)
    
    print(f"Generating {num_gen_images} images for FID calculation...")
    
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

    print("Generation complete. Calculating FID...")

    # --- 3. 计算FID ---
    try:
        metrics = calculate_metrics(
            input1=gen_dir,
            input2=real_data_path,
            cuda=(device.type == 'cuda'),
            fid=True,
            verbose=False
        )
        fid_value = metrics['frechet_inception_distance']
    except Exception as e:
        print(f"FID calculation error: {e}")
        fid_value = float('inf')
    
    # --- 4. 清理并恢复模式 ---
    shutil.rmtree(gen_dir)  # 删除临时文件夹
    generator.train()  # 恢复到训练模式
    
    return fid_value

# %% [code] {"jupyter":{"outputs_hidden":false}}
def calculate_kid(generator, real_data_path, device, latent_dim, 
                  num_gen_images=2000, eval_gen_batch_size=64):
    """
    在训练期间计算KID分数 (Kernel Inception Distance)。
    KID比FID更稳定，特别是在样本量较小的情况下。
    
    :param generator: 当前的生成器模型。
    :param real_data_path: 真实图片所在的目录。
    :param device: torch.device ('cuda' 或 'cpu')。
    :param latent_dim: 潜在向量的维度。
    :param num_gen_images: 要生成多少张图片来计算KID。
    :param eval_gen_batch_size: 生成图片时的批量大小。
    :return: (float) 计算出的KID分数（乘以1000）。
    """
    from torch_fidelity import calculate_metrics
    
    # --- 1. 创建临时文件夹保存生成的图片 ---
    gen_dir = './kid_temp_generated'
    if os.path.exists(gen_dir):
        shutil.rmtree(gen_dir)  # 清理旧的
    os.makedirs(gen_dir)
    
    print(f"Generating {num_gen_images} images for KID calculation...")
    
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

    print("Generation complete. Calculating KID...")

    # --- 3. 计算真实图像数量并确定KID子集大小 ---
    # 统计真实图像数量
    real_image_count = 0
    for file_name in os.listdir(real_data_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            real_image_count += 1
    
    # 统计生成图像数量
    gen_image_count = count
    
    # KID 子集大小必须小于两个数据集中的最小样本数
    min_samples = min(real_image_count, gen_image_count)
    # 使用最小样本数的 80% 作为 KID 子集大小（保守估计）
    kid_subset_size = max(50, int(min_samples * 0.8))  # 至少 50，最多 min_samples * 0.8
    
    print(f"Real images: {real_image_count}, Generated images: {gen_image_count}")
    print(f"Using KID subset size: {kid_subset_size}")

    # --- 4. 计算KID ---
    try:
        metrics = calculate_metrics(
            input1=gen_dir,
            input2=real_data_path,
            cuda=(device.type == 'cuda'),
            kid=True,
            kid_subset_size=kid_subset_size,  # 明确指定 KID 子集大小
            verbose=False
        )
        kid_value = metrics['kernel_inception_distance_mean'] * 1000  # KID通常乘以1000显示
    except Exception as e:
        print(f"KID calculation error: {e}")
        kid_value = float('inf')
    
    # --- 5. 清理并恢复模式 ---
    shutil.rmtree(gen_dir)  # 删除临时文件夹
    generator.train()  # 恢复到训练模式
    
    return kid_value

# %% [markdown] {"papermill":{"duration":0.037989,"end_time":"2021-10-09T06:31:39.325346","exception":false,"start_time":"2021-10-09T06:31:39.287357","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">Initializing Weights</h2>

# %% [code] {"papermill":{"duration":0.046865,"end_time":"2021-10-09T06:31:39.485146","exception":false,"start_time":"2021-10-09T06:31:39.438281","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:20.798266Z","iopub.execute_input":"2025-10-29T08:47:20.798558Z","iopub.status.idle":"2025-10-29T08:47:20.814197Z","shell.execute_reply.started":"2025-10-29T08:47:20.798529Z","shell.execute_reply":"2025-10-29T08:47:20.813407Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"id":"_gw3SMN7jtOB","papermill":{"duration":0.051132,"end_time":"2021-10-09T06:31:39.649246","exception":false,"start_time":"2021-10-09T06:31:39.598114","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:20.815040Z","iopub.execute_input":"2025-10-29T08:47:20.815280Z","iopub.status.idle":"2025-10-29T08:47:20.828509Z","shell.execute_reply.started":"2025-10-29T08:47:20.815264Z","shell.execute_reply":"2025-10-29T08:47:20.827735Z"},"jupyter":{"outputs_hidden":false}}
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

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.attention(x)  # Apply self-attention
        x = self.layer4(x)
        x = self.layer5(x)
        return x

# %% [code] {"papermill":{"duration":3.374826,"end_time":"2021-10-09T06:31:43.061517","exception":false,"start_time":"2021-10-09T06:31:39.686691","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:20.829310Z","iopub.execute_input":"2025-10-29T08:47:20.829587Z","iopub.status.idle":"2025-10-29T08:47:21.082683Z","shell.execute_reply.started":"2025-10-29T08:47:20.829566Z","shell.execute_reply":"2025-10-29T08:47:21.081948Z"},"jupyter":{"outputs_hidden":false}}
generator = Generator().to(device)
generator.apply(weights_init)
print(generator)

# %% [code] {"papermill":{"duration":0.616425,"end_time":"2021-10-09T06:31:43.756299","exception":false,"start_time":"2021-10-09T06:31:43.139874","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.083401Z","iopub.execute_input":"2025-10-29T08:47:21.083666Z","iopub.status.idle":"2025-10-29T08:47:21.389358Z","shell.execute_reply.started":"2025-10-29T08:47:21.083648Z","shell.execute_reply":"2025-10-29T08:47:21.388674Z"},"jupyter":{"outputs_hidden":false}}
summary(generator, (latent_dim,1,1))

# %% [markdown] {"papermill":{"duration":0.038779,"end_time":"2021-10-09T06:31:43.835699","exception":false,"start_time":"2021-10-09T06:31:43.79692","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">DCGAN Discriminator Network</h2>

# %% [code] {"id":"xPEMXbaJCPsQ","papermill":{"duration":0.052823,"end_time":"2021-10-09T06:31:43.927067","exception":false,"start_time":"2021-10-09T06:31:43.874244","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.390244Z","iopub.execute_input":"2025-10-29T08:47:21.390710Z","iopub.status.idle":"2025-10-29T08:47:21.396845Z","shell.execute_reply.started":"2025-10-29T08:47:21.390683Z","shell.execute_reply":"2025-10-29T08:47:21.396239Z"},"jupyter":{"outputs_hidden":false}}
class Discriminator(nn.Module):
    """
    SAGAN Discriminator (Self-Attention GAN)
    Architecture: DCGAN backbone with Self-Attention at 16x16 + Spectral Normalization
    Input: Image of size (nc, 64, 64)
    Output: Single scalar value (probability for BCE, logit for Hinge)
    Reference: Zhang et al. "Self-Attention Generative Adversarial Networks" (2018)
    """
    def __init__(self, use_sigmoid=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        
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

        # Final layer without activation (activation applied conditionally in forward)
        self.layer5_conv = nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        self.sigmoid = nn.Sigmoid()
        # State size: 1 x 1 x 1

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.attention(x)  # Apply self-attention
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5_conv(x)
        
        # Apply sigmoid only for BCE loss
        if self.use_sigmoid:
            x = self.sigmoid(x)
        
        return x.view(-1, 1).squeeze(1)

# %% [code] {"id":"1HJv-CnSkIuN","papermill":{"duration":0.071596,"end_time":"2021-10-09T06:31:44.037229","exception":false,"start_time":"2021-10-09T06:31:43.965633","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.397648Z","iopub.execute_input":"2025-10-29T08:47:21.397904Z","iopub.status.idle":"2025-10-29T08:47:21.465226Z","shell.execute_reply.started":"2025-10-29T08:47:21.397884Z","shell.execute_reply":"2025-10-29T08:47:21.464602Z"},"jupyter":{"outputs_hidden":false}}
# Initialize discriminator based on loss type
use_sigmoid = (loss_type == 'bce')
discriminator = Discriminator(use_sigmoid=use_sigmoid).to(device)
discriminator.apply(weights_init)
print(discriminator)
print(f"Discriminator output: {'with Sigmoid (BCE)' if use_sigmoid else 'without Sigmoid (Hinge)'}")

# %% [code] {"papermill":{"duration":0.0621,"end_time":"2021-10-09T06:31:44.143231","exception":false,"start_time":"2021-10-09T06:31:44.081131","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.465995Z","iopub.execute_input":"2025-10-29T08:47:21.466294Z","iopub.status.idle":"2025-10-29T08:47:21.618916Z","shell.execute_reply.started":"2025-10-29T08:47:21.466276Z","shell.execute_reply":"2025-10-29T08:47:21.618138Z"},"jupyter":{"outputs_hidden":false}}
summary(discriminator, (1,64,64))

# %% [code] {"id":"RFxQC7T0laZi","papermill":{"duration":0.045481,"end_time":"2021-10-09T06:31:44.228253","exception":false,"start_time":"2021-10-09T06:31:44.182772","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.619849Z","iopub.execute_input":"2025-10-29T08:47:21.620138Z","iopub.status.idle":"2025-10-29T08:47:21.623842Z","shell.execute_reply.started":"2025-10-29T08:47:21.620115Z","shell.execute_reply":"2025-10-29T08:47:21.623267Z"},"jupyter":{"outputs_hidden":false}}
# ============ LOSS FUNCTION SETUP ============
def discriminator_hinge_loss(real_output, fake_output):
    """
    Hinge loss for discriminator
    L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]
    """
    loss_real = torch.mean(F.relu(1.0 - real_output))
    loss_fake = torch.mean(F.relu(1.0 + fake_output))
    return loss_real + loss_fake

def generator_hinge_loss(fake_output):
    """
    Hinge loss for generator
    L_G = -E[D(G(z))]
    """
    return -torch.mean(fake_output)

# Setup loss function based on configuration
if loss_type == 'bce':
    print("Using BCE Loss (Binary Cross-Entropy)")
    print(f"  - Label Smoothing: Real={real_label_smooth}, Fake={fake_label_smooth}")
    print(f"  - Discriminator Output: Sigmoid activated (0-1)")
elif loss_type == 'hinge':
    print("Using Hinge Loss")
    print("  - No label smoothing (Hinge loss doesn't use it)")
    print("  - Discriminator Output: Raw logits")
else:
    raise ValueError(f"Unknown loss_type: {loss_type}. Use 'bce' or 'hinge'")
# ============================================

# %% [markdown] {"papermill":{"duration":0.063915,"end_time":"2021-10-09T06:31:44.635401","exception":false,"start_time":"2021-10-09T06:31:44.571486","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# ## The discriminator loss has:
# 
# - real (original images) output predictions, ground truth label as 1
# - fake (generated images) output predictions, ground truth label as 0.

# %% [code] {"papermill":{"duration":0.072788,"end_time":"2021-10-09T06:31:44.772757","exception":false,"start_time":"2021-10-09T06:31:44.699969","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.678607Z","iopub.execute_input":"2025-10-29T08:47:21.679095Z","iopub.status.idle":"2025-10-29T08:47:21.690955Z","shell.execute_reply.started":"2025-10-29T08:47:21.679077Z","shell.execute_reply":"2025-10-29T08:47:21.690370Z"},"jupyter":{"outputs_hidden":false}}
# Fixed noise for visualization
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# Labels
real_label = 1.0
fake_label = 0.0

# %% [code] {"id":"sis4zEVQkLf_","papermill":{"duration":0.077416,"end_time":"2021-10-09T06:31:44.914779","exception":false,"start_time":"2021-10-09T06:31:44.837363","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.691736Z","iopub.execute_input":"2025-10-29T08:47:21.691961Z","iopub.status.idle":"2025-10-29T08:47:21.708059Z","shell.execute_reply.started":"2025-10-29T08:47:21.691947Z","shell.execute_reply":"2025-10-29T08:47:21.707301Z"},"jupyter":{"outputs_hidden":false}}
# Setup Adam optimizers for both G and D with optimized hyperparameters
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.9), weight_decay=1e-5)
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9), weight_decay=1e-5)

# Learning rate schedulers for better convergence
# Warmup + Cosine Annealing for smoother training
schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=num_epochs, eta_min=1e-6)
schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=num_epochs, eta_min=1e-6)

# %% [markdown] {"papermill":{"duration":0.040677,"end_time":"2021-10-09T06:31:46.648014","exception":false,"start_time":"2021-10-09T06:31:46.607337","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">Training our network</h2>

# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:47:21.711536Z","iopub.execute_input":"2025-10-29T08:47:21.711839Z"},"jupyter":{"outputs_hidden":false}}
import torch
import os
from torchvision.utils import save_image

# Lists to keep track of progress
G_losses = []
D_losses = []
img_list = []
iters = 0

# FID and KID tracking
best_fid = float('inf')  # Initialize with infinity
best_kid = float('inf')  # Initialize with infinity
fid_scores = []  # Track FID scores over epochs
kid_scores = []  # Track KID scores over epochs

# Create directories for saving results
os.makedirs('./dcgan_weights', exist_ok=True)
os.makedirs('./dcgan_images', exist_ok=True)

print("Starting Optimized SAGAN (Self-Attention GAN) Training Loop...")
print(f"Architecture: DCGAN backbone + Self-Attention mechanism")
print(f"Loss Function: {loss_type.upper()} {'(Binary Cross-Entropy)' if loss_type == 'bce' else '(Hinge Loss)'}")
if loss_type == 'bce':
    print(f"  - Label Smoothing: Real={real_label_smooth}, Fake={fake_label_smooth}")
    print(f"  - Discriminator Output: Sigmoid activated (0-1)")
elif loss_type == 'hinge':
    print(f"  - No label smoothing (Hinge loss doesn't use it)")
    print(f"  - Discriminator Output: Raw logits")
print(f"\n🎯 Optimization Settings:")
print(f"  - Batch Size: {batch_size} (increased for stability)")
print(f"  - Learning Rate: {learning_rate}")
print(f"  - G Steps per D Step: {g_steps}:{d_steps}")
print(f"  - Weight Decay: 1e-5 (L2 regularization)")
print(f"  - Progressive Training: {use_progressive_gan}")
print(f"  - Total Epochs: {num_epochs}")
print(f"\nTarget: FID < 50")
print("-" * 50)

# Exponential Moving Average for Generator (improves quality)
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# Initialize EMA for generator
ema_generator = EMA(generator, decay=0.9999)

for epoch in range(num_epochs):
    epoch_d_loss = 0
    epoch_g_loss = 0
    
    # Progressive learning rate warmup
    if epoch < warmup_epochs and use_progressive_gan:
        warmup_factor = (epoch + 1) / warmup_epochs
        for param_group in optimizerD.param_groups:
            param_group['lr'] = learning_rate * warmup_factor
        for param_group in optimizerG.param_groups:
            param_group['lr'] = learning_rate * warmup_factor

    for i, (real_images, _) in enumerate(train_loader):
        ############################
        # (1) Update D network
        ###########################
        for _ in range(d_steps):
            discriminator.zero_grad()
            real_images_device = real_images.to(device)
            b_size = real_images_device.size(0)

            if loss_type == 'bce':
                # BCE Loss training
                bce_criterion = nn.BCELoss()
                # Train with real batch (with label smoothing)
                label = torch.full((b_size,), real_label_smooth, dtype=torch.float, device=device)
                output_real = discriminator(real_images_device)
                errD_real = bce_criterion(output_real, label)
                errD_real.backward()
                D_x = output_real.mean().item()

                # Train with fake batch
                noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake = generator(noise)
                label.fill_(fake_label_smooth)
                output_fake = discriminator(fake.detach())
                errD_fake = bce_criterion(output_fake, label)
                errD_fake.backward()
                D_G_z1 = output_fake.mean().item()

                errD = errD_real + errD_fake
                
            elif loss_type == 'hinge':
                # Hinge Loss training
                output_real = discriminator(real_images_device)
                noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake = generator(noise)
                output_fake = discriminator(fake.detach())
                
                errD = discriminator_hinge_loss(output_real, output_fake)
                errD.backward()
                
                D_x = output_real.mean().item()
                D_G_z1 = output_fake.mean().item()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for _ in range(g_steps):
            generator.zero_grad()
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = generator(noise)
            output = discriminator(fake)
            
            if loss_type == 'bce':
                # BCE Loss: Generator wants discriminator to think fakes are real (label = 1.0)
                bce_criterion = nn.BCELoss()
                label = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
                errG = bce_criterion(output, label)
            elif loss_type == 'hinge':
                # Hinge Loss: Maximize discriminator output
                errG = generator_hinge_loss(output)
            
            errG.backward()
            D_G_z2 = output.mean().item()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizerG.step()
            
            # Update EMA of generator weights
            ema_generator.update()

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

    # Step the learning rate schedulers (after warmup)
    if epoch >= warmup_epochs or not use_progressive_gan:
        schedulerD.step()
        schedulerG.step()

    avg_d_loss = epoch_d_loss / len(train_loader)
    avg_g_loss = epoch_g_loss / len(train_loader)
    current_lr = optimizerG.param_groups[0]['lr']
    print(f'\nEpoch {epoch} Summary: Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}, LR: {current_lr:.6f}\n')

    # Check how the generator is doing by saving G's output on fixed_noise
    if (epoch % 10 == 0) or (epoch == num_epochs-1):
        # Use EMA generator for visualization (better quality)
        ema_generator.apply_shadow()
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        ema_generator.restore()
        save_image(fake, f'./dcgan_images/fake_samples_epoch_{epoch:03d}.png',
                   normalize=True, value_range=(-1, 1), nrow=8)

    # Adaptive FID calculation frequency
    # More frequent early on, less frequent later
    should_calc_fid = False
    if epoch < 200:
        should_calc_fid = (epoch % 10 == 0 and epoch > 0)  # Every 10 epochs for first 200
    elif epoch < 800:
        should_calc_fid = (epoch % 20 == 0)  # Every 20 epochs from 200-800
    else:
        should_calc_fid = (epoch % 50 == 0)  # Every 50 epochs after 800
    
    if should_calc_fid or (epoch == num_epochs-1):
        print(f"\nCalculating FID score for epoch {epoch}...")
        # Use EMA generator for FID calculation (better quality)
        ema_generator.apply_shadow()
        current_fid = calculate_fid(
            generator=generator,
            real_data_path=real_data_path,
            device=device,
            latent_dim=latent_dim,
            num_gen_images=1000,  # More samples for better FID estimation
            eval_gen_batch_size=32
        )
        ema_generator.restore()
        fid_scores.append((epoch, current_fid))
        
        # Calculate KID score
        print(f"Calculating KID score for epoch {epoch}...")
        ema_generator.apply_shadow()
        current_kid = calculate_kid(
            generator=generator,
            real_data_path=real_data_path,
            device=device,
            latent_dim=latent_dim,
            num_gen_images=1000,
            eval_gen_batch_size=32
        )
        ema_generator.restore()
        kid_scores.append((epoch, current_kid))
        
        # Calculate improvement
        if len(fid_scores) > 1:
            prev_fid = fid_scores[-2][1]
            improvement = prev_fid - current_fid
            print(f"Epoch {epoch} - FID Score: {current_fid:.4f} (Change: {improvement:+.4f})")
        else:
            print(f"Epoch {epoch} - FID Score: {current_fid:.4f}")
        
        if len(kid_scores) > 1:
            prev_kid = kid_scores[-2][1]
            kid_improvement = prev_kid - current_kid
            print(f"Epoch {epoch} - KID Score: {current_kid:.4f} (Change: {kid_improvement:+.4f})")
        else:
            print(f"Epoch {epoch} - KID Score: {current_kid:.4f}")

        # Save model if it has the best FID score so far
        if current_fid < best_fid:
            improvement_from_best = best_fid - current_fid
            best_fid = current_fid
            print(f"🎯 New best FID score: {best_fid:.4f} (Improved by {improvement_from_best:.4f})")
            print(f"   Saving best FID model (with EMA)...")
            # Save EMA generator (better quality)
            ema_generator.apply_shadow()
            torch.save(generator.state_dict(), './dcgan_weights/generator_best_fid.pth')
            ema_generator.restore()
            torch.save(discriminator.state_dict(), './dcgan_weights/discriminator_best_fid.pth')
            # Save epoch info
            with open('./dcgan_weights/best_fid_info.txt', 'w') as f:
                f.write(f"Best FID Score: {best_fid:.4f}\n")
                f.write(f"Corresponding KID Score: {current_kid:.4f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Learning Rate: {optimizerG.param_groups[0]['lr']:.6f}\n")
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"G Steps per D Step: {g_steps}:{d_steps}\n")
        
        # Save model if it has the best KID score so far
        if current_kid < best_kid:
            kid_improvement_from_best = best_kid - current_kid
            best_kid = current_kid
            print(f"🎯 New best KID score: {best_kid:.4f} (Improved by {kid_improvement_from_best:.4f})")
            print(f"   Saving best KID model (with EMA)...")
            # Save EMA generator (better quality)
            ema_generator.apply_shadow()
            torch.save(generator.state_dict(), './dcgan_weights/generator_best_kid.pth')
            ema_generator.restore()
            torch.save(discriminator.state_dict(), './dcgan_weights/discriminator_best_kid.pth')
            # Save epoch info
            with open('./dcgan_weights/best_kid_info.txt', 'w') as f:
                f.write(f"Best KID Score: {best_kid:.4f}\n")
                f.write(f"Corresponding FID Score: {current_fid:.4f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Learning Rate: {optimizerG.param_groups[0]['lr']:.6f}\n")
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"G Steps per D Step: {g_steps}:{d_steps}\n")

    # Save model checkpoints periodically
    if (epoch % 200 == 0 and epoch > 0) or (epoch == num_epochs-1):
        # Save EMA generator checkpoint
        ema_generator.apply_shadow()
        torch.save(generator.state_dict(), f'./dcgan_weights/generator_epoch_{epoch}.pth')
        ema_generator.restore()
        torch.save(discriminator.state_dict(), f'./dcgan_weights/discriminator_epoch_{epoch}.pth')

print("Training Complete!")
print("=" * 50)
print(f"Best FID Score Achieved (vs training set): {best_fid:.4f}")
print(f"Best KID Score Achieved (vs training set): {best_kid:.4f}")
print(f"Best FID model saved at: ./dcgan_weights/generator_best_fid.pth")
print(f"Best KID model saved at: ./dcgan_weights/generator_best_kid.pth")
if best_fid < 50:
    print("🎉 SUCCESS! FID target (<50) achieved!")
elif best_fid < 60:
    print("✓ Great progress! Close to target.")
else:
    print("⚠ Consider training longer or adjusting hyperparameters.")
print("=" * 50)

# ============ FINAL EVALUATION WITH TEST SET ============
print("\n" + "=" * 50)
print("📊 FINAL EVALUATION: Testing with unseen test set")
print("=" * 50)
print(f"Test set path: {test_data_path}")
print(f"\nCalculating FID score using test set (not used during training)...")

# Calculate final FID score using test set with EMA generator
ema_generator.apply_shadow()
final_test_fid = calculate_fid(
    generator=generator,
    real_data_path=test_data_path,
    device=device,
    latent_dim=latent_dim,
    num_gen_images=222,  # Generate 222 images because test set has 222 images
    eval_gen_batch_size=32
)

# Calculate final KID score using test set with EMA generator
print(f"\nCalculating KID score using test set (not used during training)...")
final_test_kid = calculate_kid(
    generator=generator,
    real_data_path=test_data_path,
    device=device,
    latent_dim=latent_dim,
    num_gen_images=222,  # Generate 222 images because test set has 222 images
    eval_gen_batch_size=32
)
ema_generator.restore()

print("\n" + "=" * 50)
print("🎯 FINAL TEST SET RESULTS")
print("=" * 50)
print(f"FID Score (vs Test Set): {final_test_fid:.4f}")
print(f"FID Score (vs Training Set): {best_fid:.4f}")
print(f"FID Difference: {abs(final_test_fid - best_fid):.4f}")
print(f"\nKID Score (vs Test Set): {final_test_kid:.4f}")
print(f"KID Score (vs Training Set): {best_kid:.4f}")
print(f"KID Difference: {abs(final_test_kid - best_kid):.4f}")
if final_test_fid < 50:
    print("\n🎉 EXCELLENT! Test FID < 50 achieved!")
elif final_test_fid < 60:
    print("\n✓ Good performance on test set!")
elif final_test_fid < 80:
    print("\n⚠ Moderate performance. Consider more training.")
else:
    print("\n⚠ Consider adjusting hyperparameters or training longer.")
print("=" * 50)

# Save final evaluation results
with open('./dcgan_weights/final_evaluation.txt', 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("FINAL EVALUATION RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Loss Type: {loss_type.upper()}\n")
    f.write(f"Training Epochs: {num_epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Latent Dimension: {latent_dim}\n\n")
    f.write("FID Scores:\n")
    f.write(f"  - Best FID (vs Training Set): {best_fid:.4f}\n")
    f.write(f"  - Final FID (vs Test Set): {final_test_fid:.4f}\n")
    f.write(f"  - Difference: {abs(final_test_fid - best_fid):.4f}\n\n")
    f.write("KID Scores:\n")
    f.write(f"  - Best KID (vs Training Set): {best_kid:.4f}\n")
    f.write(f"  - Final KID (vs Test Set): {final_test_kid:.4f}\n")
    f.write(f"  - Difference: {abs(final_test_kid - best_kid):.4f}\n\n")
    f.write("Test Set Info:\n")
    f.write(f"  - Test Data Path: {test_data_path}\n")
    f.write(f"  - Generated Images for Evaluation: 222\n")
    f.write("=" * 50 + "\n")

print(f"\n✅ Final evaluation results saved to: ./dcgan_weights/final_evaluation.txt")

# %% [code] {"jupyter":{"outputs_hidden":false}}
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

# Subplot 2: FID Scores
if len(fid_scores) > 0:
    plt.subplot(1, 3, 2)
    epochs_fid, scores_fid = zip(*fid_scores)
    plt.title("FID Score During Training")
    plt.plot(epochs_fid, scores_fid, marker='o', color='green')
    plt.axhline(y=best_fid, color='r', linestyle='--', label=f'Best FID: {best_fid:.4f}')
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.legend()
    plt.grid(True)

# Subplot 3: KID Scores
if len(kid_scores) > 0:
    plt.subplot(1, 3, 3)
    epochs_kid, scores_kid = zip(*kid_scores)
    plt.title("KID Score During Training")
    plt.plot(epochs_kid, scores_kid, marker='s', color='blue')
    plt.axhline(y=best_kid, color='r', linestyle='--', label=f'Best KID: {best_kid:.4f}')
    plt.xlabel("Epoch")
    plt.ylabel("KID Score (x1000)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown] {"papermill":{"duration":0.174068,"end_time":"2021-10-09T14:16:51.448795","exception":false,"start_time":"2021-10-09T14:16:51.274727","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h1 style="text-align:center;font-weight: bold">Outputing Results</h1>

# %% [code] {"_kg_hide-output":true,"papermill":{"duration":0.183417,"end_time":"2021-10-09T14:16:51.808241","exception":false,"start_time":"2021-10-09T14:16:51.624824","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
def getImagePaths(path):
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names[:100]  # 只取前100个

# %% [code] {"_kg_hide-output":true,"papermill":{"duration":0.417852,"end_time":"2021-10-09T14:16:52.402334","exception":false,"start_time":"2021-10-09T14:16:51.984482","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Generate images using the best trained generator (based on FID)
generated_images_dir = './dcgan_generated'
os.makedirs(generated_images_dir, exist_ok=True)

# Load the best generator model (based on FID score)
generator_eval = Generator().to(device)
best_model_path = './dcgan_weights/generator_best_fid.pth'

# Check if best model exists, otherwise use final epoch model
if os.path.exists(best_model_path):
    print(f"Loading best model (FID: {best_fid:.4f})...")
    generator_eval.load_state_dict(torch.load(best_model_path))
else:
    print(f"Best model not found, loading final epoch model...")
    generator_eval.load_state_dict(torch.load(f'./dcgan_weights/generator_epoch_{num_epochs-1}.pth'))

generator_eval.eval()

print("Generating images...")
num_images = 600
with torch.no_grad():
    for i in range(num_images):
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        fake_image = generator_eval(noise)
        save_path = os.path.join(generated_images_dir, f'generated_{i:04d}.png')
        # 明确指定 value_range=(-1, 1)，因为生成器使用 Tanh() 激活
        save_image(fake_image, save_path, normalize=True, value_range=(-1, 1))

print(f"Generated {num_images} images in {generated_images_dir}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
display_multiple_img(getImagePaths(generated_images_dir))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Summary
# 
# This script implements a standard DCGAN with the following key components:
# 
# 1. **Generator**: Uses transposed convolutions to upsample from a latent vector (100-dim) to a 64x64 image
# 2. **Discriminator**: Uses strided convolutions to downsample a 64x64 image to a single probability score
# 3. **Training**: Uses Binary Cross-Entropy loss for both networks
# 4. **Architecture Guidelines** (following DCGAN paper):
#    - Replace pooling layers with strided convolutions
#    - Use batch normalization in both networks
#    - Remove fully connected hidden layers
#    - Use ReLU activation in generator (except output layer which uses Tanh)
#    - Use LeakyReLU activation in discriminator