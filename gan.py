# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:45:36.175246Z","iopub.execute_input":"2025-10-29T08:45:36.175473Z","iopub.status.idle":"2025-10-29T08:47:17.862566Z","shell.execute_reply.started":"2025-10-29T08:45:36.175428Z","shell.execute_reply":"2025-10-29T08:47:17.861917Z"},"jupyter":{"outputs_hidden":false}}
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
from pytorch_fid import fid_score  # 导入FID计算模块
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
# ## DCGAN 参数配置

# %% [code] {"id":"xVXC_q2ekuf8","papermill":{"duration":0.079089,"end_time":"2021-10-09T06:31:24.988503","exception":false,"start_time":"2021-10-09T06:31:24.909414","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:17.864024Z","iopub.execute_input":"2025-10-29T08:47:17.864499Z","iopub.status.idle":"2025-10-29T08:47:17.926536Z","shell.execute_reply.started":"2025-10-29T08:47:17.864478Z","shell.execute_reply":"2025-10-29T08:47:17.925750Z"},"jupyter":{"outputs_hidden":false}}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8  # Smaller batch for small dataset
learning_rate = 0.0001  # Reduced learning rate for stability
num_epochs = 2000  # More epochs for small dataset
image_size = 64  # Image size (64x64)
latent_dim = 128  # Increased latent dimension for more diversity
nc = 1  # Number of channels (1 for grayscale, 3 for RGB)
ngf = 64  # Number of generator filters
ndf = 64  # Number of discriminator filters
# Training strategy parameters
d_steps = 1  # Discriminator steps per generator step
g_steps = 2  # Generator steps (train G more often)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## data prepare

# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:47:17.927412Z","iopub.execute_input":"2025-10-29T08:47:17.927751Z","iopub.status.idle":"2025-10-29T08:47:19.348568Z","shell.execute_reply.started":"2025-10-29T08:47:17.927723Z","shell.execute_reply":"2025-10-29T08:47:19.347926Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:47:19.349317Z","iopub.execute_input":"2025-10-29T08:47:19.349601Z","iopub.status.idle":"2025-10-29T08:47:19.951009Z","shell.execute_reply.started":"2025-10-29T08:47:19.349571Z","shell.execute_reply":"2025-10-29T08:47:19.950319Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"id":"rmKRUtX6kwt1","papermill":{"duration":12.148366,"end_time":"2021-10-09T06:31:37.214299","exception":false,"start_time":"2021-10-09T06:31:25.065933","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:19.951886Z","iopub.execute_input":"2025-10-29T08:47:19.952156Z","iopub.status.idle":"2025-10-29T08:47:19.960161Z","shell.execute_reply.started":"2025-10-29T08:47:19.952132Z","shell.execute_reply":"2025-10-29T08:47:19.959481Z"},"jupyter":{"outputs_hidden":false}}
# Optimized data transforms - minimal augmentation for better quality
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=nc),
    # Minimal augmentation - only flips for better FID
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
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
                  num_gen_images=2000, eval_gen_batch_size=64, 
                  fid_calc_batch_size=50, dims=2048):
    """
    在训练期间计算FID分数。
    
    :param generator: 当前的生成器模型。
    :param real_data_path: 真实图片所在的目录 (例如 './train/data')。
    :param device: torch.device ('cuda' 或 'cpu')。
    :param latent_dim: 潜在向量的维度。
    :param num_gen_images: 要生成多少张图片来计算FID。
    :param eval_gen_batch_size: 生成图片时的批量大小。
    :param fid_calc_batch_size: FID计算器内部的批量大小。
    :param dims: Inception V3 模型的特征维度 (2048 是标准)。
    :return: (float) 计算出的FID分数。
    """
    
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
            for i in range(current_batch_size):
                save_path = os.path.join(gen_dir, f'img_{count + i}.png')
                save_image(fake_images[i], save_path, normalize=True)
            
            count += current_batch_size

    print("Generation complete. Calculating FID...")

    # --- 3. 计算FID ---
    # 定义两个路径
    paths = [real_data_path, gen_dir]
    
    # 调用 pytorch-fid 模块
    fid_value = fid_score.calculate_fid_given_paths(
        paths=paths,
        batch_size=fid_calc_batch_size,
        device=device,
        dims=dims
    )
    
    # --- 4. 清理并恢复模式 ---
    shutil.rmtree(gen_dir)  # 删除临时文件夹
    generator.train()  # 恢复到训练模式
    
    return fid_value

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
    Improved DCGAN Generator with Self-Attention
    Input: latent vector z of dimension (latent_dim, 1, 1)
    Output: Generated image of size (nc, 64, 64)
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
    Improved DCGAN Discriminator with Self-Attention and Spectral Normalization
    Input: Image of size (nc, 64, 64)
    Output: Single scalar value (probability of being real)
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
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
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

# %% [code] {"id":"1HJv-CnSkIuN","papermill":{"duration":0.071596,"end_time":"2021-10-09T06:31:44.037229","exception":false,"start_time":"2021-10-09T06:31:43.965633","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.397648Z","iopub.execute_input":"2025-10-29T08:47:21.397904Z","iopub.status.idle":"2025-10-29T08:47:21.465226Z","shell.execute_reply.started":"2025-10-29T08:47:21.397884Z","shell.execute_reply":"2025-10-29T08:47:21.464602Z"},"jupyter":{"outputs_hidden":false}}
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)

# %% [code] {"papermill":{"duration":0.0621,"end_time":"2021-10-09T06:31:44.143231","exception":false,"start_time":"2021-10-09T06:31:44.081131","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.465995Z","iopub.execute_input":"2025-10-29T08:47:21.466294Z","iopub.status.idle":"2025-10-29T08:47:21.618916Z","shell.execute_reply.started":"2025-10-29T08:47:21.466276Z","shell.execute_reply":"2025-10-29T08:47:21.618138Z"},"jupyter":{"outputs_hidden":false}}
summary(discriminator, (1,64,64))

# %% [code] {"id":"RFxQC7T0laZi","papermill":{"duration":0.045481,"end_time":"2021-10-09T06:31:44.228253","exception":false,"start_time":"2021-10-09T06:31:44.182772","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.619849Z","iopub.execute_input":"2025-10-29T08:47:21.620138Z","iopub.status.idle":"2025-10-29T08:47:21.623842Z","shell.execute_reply.started":"2025-10-29T08:47:21.620115Z","shell.execute_reply":"2025-10-29T08:47:21.623267Z"},"jupyter":{"outputs_hidden":false}}
# Binary Cross Entropy Loss for DCGAN
criterion = nn.BCELoss()

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
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

# Learning rate schedulers for better convergence
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

# FID tracking
best_fid = float('inf')  # Initialize with infinity
fid_scores = []  # Track FID scores over epochs

# Create directories for saving results
os.makedirs('./dcgan_weights', exist_ok=True)
os.makedirs('./dcgan_images', exist_ok=True)

print("Starting Improved DCGAN Training Loop...")
print("-" * 50)

for epoch in range(num_epochs):
    epoch_d_loss = 0
    epoch_g_loss = 0

    for i, (real_images, _) in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        for _ in range(d_steps):
            discriminator.zero_grad()
            real_images_device = real_images.to(device)
            b_size = real_images_device.size(0)

            # Train with real batch
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_images_device)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake batch
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        for _ in range(g_steps):
            generator.zero_grad()
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(real_label)
            output = discriminator(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
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
                   normalize=True, nrow=8)

    # Calculate FID score every 20 epochs (less frequent to save time)
    if (epoch % 20 == 0 and epoch > 0) or (epoch == num_epochs-1):
        print(f"\nCalculating FID score for epoch {epoch}...")
        current_fid = calculate_fid(
            generator=generator,
            real_data_path=real_data_path,
            device=device,
            latent_dim=latent_dim,
            num_gen_images=300,  # Match dataset size
            eval_gen_batch_size=32,
            fid_calc_batch_size=50,
            dims=2048
        )
        fid_scores.append((epoch, current_fid))
        print(f"Epoch {epoch} - FID Score: {current_fid:.4f}")

        # Save model if it has the best FID score so far
        if current_fid < best_fid:
            best_fid = current_fid
            print(f"New best FID score: {best_fid:.4f} - Saving best model...")
            torch.save(generator.state_dict(), './dcgan_weights/generator_best_fid.pth')
            torch.save(discriminator.state_dict(), './dcgan_weights/discriminator_best_fid.pth')
            # Save epoch info
            with open('./dcgan_weights/best_fid_info.txt', 'w') as f:
                f.write(f"Best FID Score: {best_fid:.4f}\n")
                f.write(f"Epoch: {epoch}\n")

    # Save model checkpoints periodically
    if (epoch % 100 == 0 and epoch > 0) or (epoch == num_epochs-1):
        torch.save(generator.state_dict(), f'./dcgan_weights/generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'./dcgan_weights/discriminator_epoch_{epoch}.pth')

print("Training Complete!")
print(f"\nBest FID Score: {best_fid:.4f}")
print(f"Best model saved at: ./dcgan_weights/generator_best_fid.pth")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Plot the training losses
plt.figure(figsize=(15, 5))

# Subplot 1: Losses
plt.subplot(1, 2, 1)
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Subplot 2: FID Scores
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
num_images = 100
with torch.no_grad():
    for i in range(num_images):
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        fake_image = generator_eval(noise)
        save_path = os.path.join(generated_images_dir, f'generated_{i:04d}.png')
        save_image(fake_image, save_path, normalize=True)

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