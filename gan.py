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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.manual_seed(1)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 参数配置

# %% [code] {"id":"xVXC_q2ekuf8","papermill":{"duration":0.079089,"end_time":"2021-10-09T06:31:24.988503","exception":false,"start_time":"2021-10-09T06:31:24.909414","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:17.864024Z","iopub.execute_input":"2025-10-29T08:47:17.864499Z","iopub.status.idle":"2025-10-29T08:47:17.926536Z","shell.execute_reply.started":"2025-10-29T08:47:17.864478Z","shell.execute_reply":"2025-10-29T08:47:17.925750Z"},"jupyter":{"outputs_hidden":false}}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
learning_rate = 0.0002 
num_epochs = 1500
EVAL_FREQ = 50  # 每 50 个 epoch 评估一次
best_fid = float('inf')  # 初始化为无穷大
best_fid_epoch = 0
n_critic = 5  # 设置 D 的训练次数
image_shape = (3, 64, 64)
image_dim = int(np.prod(image_shape))
latent_dim = 128

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
# ## Data Augment

# %% [code] {"id":"rmKRUtX6kwt1","papermill":{"duration":12.148366,"end_time":"2021-10-09T06:31:37.214299","exception":false,"start_time":"2021-10-09T06:31:25.065933","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:19.951886Z","iopub.execute_input":"2025-10-29T08:47:19.952156Z","iopub.status.idle":"2025-10-29T08:47:19.960161Z","shell.execute_reply.started":"2025-10-29T08:47:19.952132Z","shell.execute_reply":"2025-10-29T08:47:19.959481Z"},"jupyter":{"outputs_hidden":false}}
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
# 1. 创建一个自定义变换类，用于实现 (0, 90, 180, 270) 度的随机旋转
class RandomDiscreteRotation:
    """
    应用 0, 90, 180, 270 度中的一个随机旋转。
    这些是无伪影的旋转（像素重排）。
    """
    def __init__(self):
        # 旋转角度必须是 [0, 90, 180, 270]
        # 注意：TF.rotate 使用的是逆时针旋转
        self.angles = [0, 90, 180, 270] 

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): 要被旋转的图像。
        Returns:
            PIL Image or Tensor: 旋转后的图像。
        """
        # 从列表中随机选择一个角度
        angle = random.choice(self.angles)
        
        # TF.rotate 会自动处理 PIL 图像或 Tensor
        # 重要的是，对于 90, 180, 270 的旋转，这是无损的
        return TF.rotate(img, angle)
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1), 
    # --- 新增的对称群 ---
    # 1. 首先，应用 (0, 90, 180, 270) 度中的一个随机旋转
    RandomDiscreteRotation(),
    
    # 2. 接着，应用 50% 的概率进行随机水平翻转
    transforms.RandomHorizontalFlip(p=0.5),  # 添加随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_dataset = datasets.ImageFolder(root='./train', transform=train_transform) #原始为../input/efficientnet-data/test/
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)

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
# <h2 style="text-align:center;font-weight: bold;">Generator Network</h2>

# %% [code] {"id":"_gw3SMN7jtOB","papermill":{"duration":0.051132,"end_time":"2021-10-09T06:31:39.649246","exception":false,"start_time":"2021-10-09T06:31:39.598114","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:20.815040Z","iopub.execute_input":"2025-10-29T08:47:20.815280Z","iopub.status.idle":"2025-10-29T08:47:20.828509Z","shell.execute_reply.started":"2025-10-29T08:47:20.815264Z","shell.execute_reply":"2025-10-29T08:47:20.827735Z"},"jupyter":{"outputs_hidden":false}}
import torchvision.models as models
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output    
        
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         # 主路径：上采样 + 卷积块
#         self.block = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 上采样
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#         # 跳跃连接：直接上采样 + 1x1 卷积
#         self.shortcut = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )

#     def forward(self, x):
#         shortcut = self.shortcut(x)  # 跳跃连接
#         block_output = self.block(x)  # 主路径
#         return F.relu(block_output + shortcut, inplace=True)  # 输出

# class Generator(nn.Module):
#     """
#     基于 ResNet 的生成器，生成分辨率为 64x64 的图像。
#     """
#     def __init__(self, latent_dim=128):
#         super(Generator, self).__init__()
#         # 初始映射层
#         self.initial = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True)
#         )
#         # 残差块
#         self.res_blocks = nn.Sequential(
#             ResidualBlock(512, 256),  # 4x4 -> 8x8
#             ResidualBlock(256, 128),  # 8x8 -> 16x16
#             ResidualBlock(128, 64)   # 16x16 -> 32x32
#         )
#         # 输出层
#         self.final_conv = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)  # 32x32 -> 64x64
#         self.final_activation = nn.Tanh()

#     def forward(self, z):
#         x = self.initial(z)
#         x = self.res_blocks(x)
#         x = self.final_conv(x)
#         x = self.final_activation(x)
#         return x
# class Generator(nn.Module):
#     def __init__(self, latent_dim):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             # 输入: latent_dim x 1 x 1 -> 64 * 8 x 4 x 4
#             nn.ConvTranspose2d(latent_dim, 64 * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(64 * 8),
#             nn.ReLU(True),
#             # 64 * 8 x 4 x 4 -> 64 * 4 x 8 x 8
#             nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 4),
#             nn.ReLU(True),
#             # 64 * 4 x 8 x 8 -> 64 * 2 x 16 x 16
#             nn.ConvTranspose2d(64 * 4, 1, 4, 2, 1, bias=False),
#             nn.Tanh()  # 将输出归一化到 [-1, 1]
#         )

#     def forward(self, input):
#         return self.main(input)

# %% [code] {"papermill":{"duration":3.374826,"end_time":"2021-10-09T06:31:43.061517","exception":false,"start_time":"2021-10-09T06:31:39.686691","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:20.829310Z","iopub.execute_input":"2025-10-29T08:47:20.829587Z","iopub.status.idle":"2025-10-29T08:47:21.082683Z","shell.execute_reply.started":"2025-10-29T08:47:20.829566Z","shell.execute_reply":"2025-10-29T08:47:21.081948Z"},"jupyter":{"outputs_hidden":false}}
generator = Generator().to(device)
generator.apply(weights_init)
print(generator)

# %% [code] {"papermill":{"duration":0.616425,"end_time":"2021-10-09T06:31:43.756299","exception":false,"start_time":"2021-10-09T06:31:43.139874","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.083401Z","iopub.execute_input":"2025-10-29T08:47:21.083666Z","iopub.status.idle":"2025-10-29T08:47:21.389358Z","shell.execute_reply.started":"2025-10-29T08:47:21.083648Z","shell.execute_reply":"2025-10-29T08:47:21.388674Z"},"jupyter":{"outputs_hidden":false}}
summary(generator, (latent_dim,1,1))

# %% [markdown] {"papermill":{"duration":0.038779,"end_time":"2021-10-09T06:31:43.835699","exception":false,"start_time":"2021-10-09T06:31:43.79692","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">Descriminator Network</h2>

# %% [code] {"id":"xPEMXbaJCPsQ","papermill":{"duration":0.052823,"end_time":"2021-10-09T06:31:43.927067","exception":false,"start_time":"2021-10-09T06:31:43.874244","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.390244Z","iopub.execute_input":"2025-10-29T08:47:21.390710Z","iopub.status.idle":"2025-10-29T08:47:21.396845Z","shell.execute_reply.started":"2025-10-29T08:47:21.390683Z","shell.execute_reply":"2025-10-29T08:47:21.396239Z"},"jupyter":{"outputs_hidden":false}}
from torch.nn.utils import spectral_norm

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             # 移除了 spectral_norm 包装
#             nn.Conv2d(3, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # 移除了 spectral_norm 包装
#             nn.Conv2d(64, 128, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # 移除了 spectral_norm 包装
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # 移除了 spectral_norm 包装
#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # 最后一层没有 spectral_norm，保持不变
#             nn.Conv2d(512, 1, 4, 1, 0, bias=False),
#             nn.Flatten()
#         )

#     def forward(self, input):
#         output = self.main(input)
#         return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # nn.Conv2d(3, 64, 4, 2, 1, bias=False), # <-- 4. 注释掉老的
            # 输入: 1 x 64 x 64
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 2x2
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 2x2 -> 1x1
            nn.Conv2d(256, 512, 2, 1, 0, bias=False),  # 修改 kernel_size 为 2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出标量
            nn.Conv2d(512, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)

# %% [code] {"id":"1HJv-CnSkIuN","papermill":{"duration":0.071596,"end_time":"2021-10-09T06:31:44.037229","exception":false,"start_time":"2021-10-09T06:31:43.965633","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.397648Z","iopub.execute_input":"2025-10-29T08:47:21.397904Z","iopub.status.idle":"2025-10-29T08:47:21.465226Z","shell.execute_reply.started":"2025-10-29T08:47:21.397884Z","shell.execute_reply":"2025-10-29T08:47:21.464602Z"},"jupyter":{"outputs_hidden":false}}
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)

# %% [code] {"papermill":{"duration":0.0621,"end_time":"2021-10-09T06:31:44.143231","exception":false,"start_time":"2021-10-09T06:31:44.081131","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.465995Z","iopub.execute_input":"2025-10-29T08:47:21.466294Z","iopub.status.idle":"2025-10-29T08:47:21.618916Z","shell.execute_reply.started":"2025-10-29T08:47:21.466276Z","shell.execute_reply":"2025-10-29T08:47:21.618138Z"},"jupyter":{"outputs_hidden":false}}
summary(discriminator, (1,64,64))

# %% [code] {"id":"RFxQC7T0laZi","papermill":{"duration":0.045481,"end_time":"2021-10-09T06:31:44.228253","exception":false,"start_time":"2021-10-09T06:31:44.182772","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.619849Z","iopub.execute_input":"2025-10-29T08:47:21.620138Z","iopub.status.idle":"2025-10-29T08:47:21.623842Z","shell.execute_reply.started":"2025-10-29T08:47:21.620115Z","shell.execute_reply":"2025-10-29T08:47:21.623267Z"},"jupyter":{"outputs_hidden":false}}
adversarial_loss = nn.BCELoss()

# %% [code] {"papermill":{"duration":0.046164,"end_time":"2021-10-09T06:31:44.312994","exception":false,"start_time":"2021-10-09T06:31:44.26683","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.624613Z","iopub.execute_input":"2025-10-29T08:47:21.624869Z","iopub.status.idle":"2025-10-29T08:47:21.636620Z","shell.execute_reply.started":"2025-10-29T08:47:21.624848Z","shell.execute_reply":"2025-10-29T08:47:21.635950Z"},"jupyter":{"outputs_hidden":false}}
def generator_loss(fake_output, label):
    gen_loss = adversarial_loss(fake_output, label)
    return gen_loss

# %% [code] {"papermill":{"duration":0.074971,"end_time":"2021-10-09T06:31:44.503934","exception":false,"start_time":"2021-10-09T06:31:44.428963","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.637317Z","iopub.execute_input":"2025-10-29T08:47:21.637616Z","iopub.status.idle":"2025-10-29T08:47:21.650409Z","shell.execute_reply.started":"2025-10-29T08:47:21.637589Z","shell.execute_reply":"2025-10-29T08:47:21.649661Z"},"jupyter":{"outputs_hidden":false}}
## The generator_loss function is fed two parameters:

# - fake_output: Output predictions from the discriminator, when fed generator-produced images.
# - label: Ground truth labels (1), for you would like the generator to fool the discriminator and produce real images. Hence, the labels would be one.
def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:47:21.651136Z","iopub.execute_input":"2025-10-29T08:47:21.651494Z","iopub.status.idle":"2025-10-29T08:47:21.663583Z","shell.execute_reply.started":"2025-10-29T08:47:21.651467Z","shell.execute_reply":"2025-10-29T08:47:21.662929Z"},"jupyter":{"outputs_hidden":false}}
import torch.nn.functional as F  # 导入 F 模块

def discriminator_loss(real_output, fake_output):
    return torch.mean(F.relu(1.0 - real_output)) + torch.mean(F.relu(1.0 + fake_output))

def generator_loss(fake_output,label=None):
    return -torch.mean(fake_output)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:47:21.664393Z","iopub.execute_input":"2025-10-29T08:47:21.664692Z","iopub.status.idle":"2025-10-29T08:47:21.677681Z","shell.execute_reply.started":"2025-10-29T08:47:21.664668Z","shell.execute_reply":"2025-10-29T08:47:21.676946Z"},"jupyter":{"outputs_hidden":false}}
def gradient_penalty(critic, real_data, fake_data, device):
    """
    计算梯度惩罚项，确保判别器满足 Lipschitz 连续性。
    :param critic: 判别器模型
    :param real_data: 真实样本
    :param fake_data: 生成样本
    :param device: 当前设备
    :return: 梯度惩罚值
    """
    # 生成插值样本
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)  # 随机权重
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    # 判别器对插值样本的输出
    critic_output = critic(interpolates)

    # 计算插值样本的梯度
    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_output, device=device),
        create_graph=True,
        retain_graph=True
    )[0]

    # 计算梯度范数和惩罚
    gradients = gradients.view(batch_size, -1)  # 展平
    gradient_norm = gradients.norm(2, dim=1)  # 计算 L2 范数
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

# %% [markdown] {"papermill":{"duration":0.063915,"end_time":"2021-10-09T06:31:44.635401","exception":false,"start_time":"2021-10-09T06:31:44.571486","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# ## The discriminator loss has:
# 
# - real (original images) output predictions, ground truth label as 1
# - fake (generated images) output predictions, ground truth label as 0.

# %% [code] {"papermill":{"duration":0.072788,"end_time":"2021-10-09T06:31:44.772757","exception":false,"start_time":"2021-10-09T06:31:44.699969","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.678607Z","iopub.execute_input":"2025-10-29T08:47:21.679095Z","iopub.status.idle":"2025-10-29T08:47:21.690955Z","shell.execute_reply.started":"2025-10-29T08:47:21.679077Z","shell.execute_reply":"2025-10-29T08:47:21.690370Z"},"jupyter":{"outputs_hidden":false}}
fixed_noise = torch.randn(128, latent_dim, 1, 1, device=device)
real_label = 1
fake_label = 0

# %% [code] {"id":"sis4zEVQkLf_","papermill":{"duration":0.077416,"end_time":"2021-10-09T06:31:44.914779","exception":false,"start_time":"2021-10-09T06:31:44.837363","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2025-10-29T08:47:21.691736Z","iopub.execute_input":"2025-10-29T08:47:21.691961Z","iopub.status.idle":"2025-10-29T08:47:21.708059Z","shell.execute_reply.started":"2025-10-29T08:47:21.691947Z","shell.execute_reply":"2025-10-29T08:47:21.707301Z"},"jupyter":{"outputs_hidden":false}}
G_optimizer = optim.Adam(generator.parameters(), lr = learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate, betas=(0.5, 0.999)) # 单个粒子使用0.3就可以了，分离使用0.04 单个粒子0.03

# %% [markdown] {"papermill":{"duration":0.040677,"end_time":"2021-10-09T06:31:46.648014","exception":false,"start_time":"2021-10-09T06:31:46.607337","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# 
# <h2 style="text-align:center;font-weight: bold;">Training our network</h2>

# %% [code] {"execution":{"iopub.status.busy":"2025-10-29T08:47:21.711536Z","iopub.execute_input":"2025-10-29T08:47:21.711839Z"},"jupyter":{"outputs_hidden":false}}
import torch
import os
from torchvision.utils import save_image
import torch.nn.functional as F  # 用于 ReLU

# 初始化
D_loss_plot, G_loss_plot = [], []
best_loss_diff = float('inf')  # 初始化为无穷大

# 定义保存路径
os.makedirs('./t_weights', exist_ok=True)
os.makedirs('./images', exist_ok=True)

for epoch in range(1, num_epochs + 1):
    D_loss_list, G_loss_list = [], []

    for index, (real_images, _) in enumerate(train_loader):
        # ==================== 判别器训练 ====================
        D_optimizer.zero_grad()
        real_images = real_images.to(device)

        # 生成假样本
        noise_vector = torch.randn(real_images.size(0), latent_dim, 1, 1, device=device)
        generated_image = generator(noise_vector)

        # 判别器输出
        real_output = discriminator(real_images)
        fake_output = discriminator(generated_image.detach())

        # 梯度惩罚
        lambda_gp = 10  # 梯度惩罚权重
        gp = gradient_penalty(discriminator, real_images, generated_image.detach(), device)

        # 判别器损失（WGAN-GP）
        D_loss = -torch.mean(real_output) + torch.mean(fake_output) + lambda_gp * gp
        D_loss.backward()
        D_loss_list.append(D_loss.item())
        D_optimizer.step()
        # ==================== 生成器训练 ====================
        if (index + 1) % n_critic == 0:
            G_optimizer.zero_grad()
    
            # 生成器的假样本输出
            fake_output = discriminator(generated_image)
    
            # 生成器损失
            G_loss = -torch.mean(fake_output)
            G_loss.backward()
            G_loss_list.append(G_loss.item())
            G_optimizer.step()
        
    # --- 【！！！在这里添加修正！！！】 ---
    # 计算这个 epoch 的平均损失并添加到主列表
    # （注意：G_loss_list 可能为空，如果 n_critic > len(train_loader)）
    epoch_d_loss = np.mean(D_loss_list) if D_loss_list else 0
    epoch_g_loss = np.mean(G_loss_list) if G_loss_list else 0
    
    D_loss_plot.append(epoch_d_loss)
    G_loss_plot.append(epoch_g_loss)
    
    if epoch % EVAL_FREQ == 0 and epoch > 0:
        print(f"\n--- [Epoch {epoch}] Starting FID evaluation ---")
        
        # 调用我们创建的函数
        current_fid = calculate_fid(
            generator=generator,
            real_data_path=real_data_path,
            device=device,
            latent_dim=latent_dim,
            num_gen_images=1000  # 评估时生成1000张。
        )
        
        print(f"--- [Epoch {epoch}] FID Score: {current_fid:.4f} ---")
        
        # 检查这是否是目前最好的FID
        if current_fid < best_fid:
            best_fid = current_fid
            best_fid_epoch = epoch
            print(f"!!! New Best FID: {best_fid:.4f} at Epoch {epoch}. Saving model. !!!\n")
            
            # 保存最佳模型
            torch.save(generator.state_dict(), './t_weights/best_generator_fid.pth')
            torch.save(discriminator.state_dict(), './t_weights/best_discriminator_fid.pth')
        else:
            print(f"FID did not improve. Best FID is still {best_fid:.4f} from epoch {best_fid_epoch}.\n")

# 最后保存损失曲线
torch.save(D_loss_plot, './t_weights/D_loss_plot.pth')
torch.save(G_loss_plot, './t_weights/G_loss_plot.pth')

# %% [code] {"jupyter":{"outputs_hidden":false}}
 #绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(D_loss_plot, label='Discriminator Loss (D)', color='red', linewidth=2)
plt.plot(G_loss_plot, label='Generator Loss (G)', color='blue', linewidth=2)

# 添加标题和标签
plt.title('GAN Training Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)

# 添加图例
plt.legend(fontsize=12)
plt.grid(True)

loss_diff = [abs(d - g) for d, g in zip(D_loss_plot, G_loss_plot)]
plt.plot(loss_diff, label='Loss Difference', color='green', linewidth=2)

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

# %% [code] {"_kg_hide-output":true,"papermill":{"duration":32.195267,"end_time":"2021-10-09T14:17:24.773111","exception":false,"start_time":"2021-10-09T14:16:52.577844","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# display_multiple_img(getImagePaths('./images'))

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

real_images_dir = './train'
generated_images_dir = './generated_images'
os.makedirs(real_images_dir, exist_ok=True)
os.makedirs(generated_images_dir, exist_ok=True)

generator = Generator()  # 使用你定义的生成器类
# generator.load_state_dict(torch.load(f'./t_weights/generator_epoch_{num_epochs}.pth'))  # 加载最后一轮的生成器权重
# generator.load_state_dict(torch.load(f'./t_weights/best_generator.pth'))
generator.load_state_dict(torch.load(f'./t_weights/best_generator_fid.pth'))

generator.eval()

def generate_images(generator, num_images, latent_dim, save_dir):
    generator.eval()
    noise_vector = torch.randn(num_images, latent_dim, 1, 1)
    with torch.no_grad():
        generated_images = generator(noise_vector)
    # generated_images = torch.nn.functional.interpolate(generated_images, size=(64, 64))  
    
    # 保存每张图像到指定目录
    for i in range(num_images):
        save_path = os.path.join(save_dir, f'generated_image_{i + 1:03d}.png')
        save_image(generated_images[i], save_path, normalize=True)


def save_real_images(dataset_path, save_dir, num_images=50):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    real_dataset = ImageFolder(root=dataset_path, transform=transform)
    real_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)

    # 保存前 num_images 的真实图像
    for i, (image, _) in enumerate(real_loader):
        if i >= num_images:
            break
        save_path = os.path.join(save_dir, f'real_image_{i + 1:03d}.png')
        save_image(image[0], save_path, normalize=True)

generate_images(generator, num_images=200, latent_dim=latent_dim, save_dir=generated_images_dir)

path_to_real_dataset = './real_images'
os.makedirs(path_to_real_dataset,exist_ok=True)
save_real_images(real_images_dir, path_to_real_dataset, num_images=200)

# 提示用户使用 pytorch-fid 工具
print(f"Run the following command to compute FID:")
print(f"pytorch-fid {path_to_real_dataset} {generated_images_dir}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
display_multiple_img(getImagePaths('./generated_images'))

# %% [code] {"jupyter":{"outputs_hidden":false}}
!python -m pytorch_fid  ./real_images ./generated_images

# %% [code] {"jupyter":{"outputs_hidden":false}}
generate_images(generator, num_images=500, latent_dim=latent_dim, save_dir=generated_images_dir)

# %% [code] {"jupyter":{"outputs_hidden":false}}
!zip -qr generated_images.zip ./generated_images

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 理论FID下限

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os
import shutil
import random

def split_dataset_for_intra_fid(source_dir, dest_dir_A, dest_dir_B):
    """
    将源目录中的图片随机分成两半，拷贝到两个新目录中。
    """
    
    # 1. 定义图片扩展名
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    # 2. 清理并创建目标目录
    for dir_path in [dest_dir_A, dest_dir_B]:
        if os.path.exists(dir_path):
            print(f"正在清理已存在的目录: {dir_path}")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        print(f"已创建空目录: {dir_path}")

    # 3. 收集所有图片
    try:
        all_image_files = [
            f for f in os.listdir(source_dir) 
            if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(image_extensions)
        ]
    except FileNotFoundError:
        print(f"错误：源目录未找到: {source_dir}")
        print("请确认 'source_dir' 路径是否正确。")
        return

    if not all_image_files:
        print(f"错误：在 {source_dir} 中没有找到任何图片文件。")
        return
        
    print(f"在 {source_dir} 中共找到 {len(all_image_files)} 张图片。")
    
    # 4. 随机打乱列表
    random.shuffle(all_image_files)
    
    # 5. 找到分割点
    split_point = len(all_image_files) // 2
    files_A = all_image_files[:split_point]
    files_B = all_image_files[split_point:]
    
    print(f"将分割为: 集合A ({len(files_A)} 张) 和 集合B ({len(files_B)} 张)")
    
    # 6. 拷贝文件到 A
    print(f"正在拷贝文件到 {dest_dir_A}...")
    for file_name in files_A:
        src_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir_A, file_name)
        shutil.copy(src_path, dest_path)
        
    # 7. 拷贝文件到 B
    print(f"正在拷贝文件到 {dest_dir_B}...")
    for file_name in files_B:
        src_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir_B, file_name)
        shutil.copy(src_path, dest_path)
        
    print("\n--- 拷贝完成！---")



# 两个目标目录
DEST_A_DIR = './real_set_A'
DEST_B_DIR = './real_set_B'

# 运行分割函数
split_dataset_for_intra_fid(real_data_path, DEST_A_DIR, DEST_B_DIR)
!python -m pytorch_fid ./real_set_A ./real_set_B

# %% [code] {"jupyter":{"outputs_hidden":false}}
