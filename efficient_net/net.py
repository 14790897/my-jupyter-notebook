# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:27:42.764355Z","iopub.execute_input":"2025-10-28T14:27:42.764569Z","iopub.status.idle":"2025-10-28T14:27:42.776652Z","shell.execute_reply.started":"2025-10-28T14:27:42.764544Z","shell.execute_reply":"2025-10-28T14:27:42.775942Z"},"jupyter":{"outputs_hidden":false}}
import shutil
import time

from PIL import Image, ImageOps

fixed_seed = 1028
# ===== 配置选项：是否使用GAN生成的数据 =====
USE_GAN_DATA = True  # 设置为 False 则只使用真实数据，True 则添加GAN数据

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:27:42.778281Z","iopub.execute_input":"2025-10-28T14:27:42.778552Z","iopub.status.idle":"2025-10-28T14:27:42.787551Z","shell.execute_reply.started":"2025-10-28T14:27:42.778526Z","shell.execute_reply":"2025-10-28T14:27:42.786731Z"},"jupyter":{"outputs_hidden":false}}
import os
from pathlib import Path


def copy_files_with_prefix(source_map, destination_dir, recursive=False):
    """
    从多个源目录复制文件到一个目标目录，并为每个源的文件名添加唯一的前缀。

    参数:
    source_map (dict):
        一个字典，键 (key) 是您想要添加的前缀 (str)，
        值 (value) 是源文件夹的路径 (str 或 Path)。
        示例: {'gen1': '/path/to/images1', 'gen2': '/path/to/images2'}

    destination_dir (str 或 Path):
        所有文件将被复制到的那一个目标文件夹。
        示例: '/kaggle/working/my_data/0'

    recursive (bool, 可选):
        是否递归搜索源文件夹中的子目录。
        - False (默认): 仅复制源文件夹根目录下的文件。
        - True: 递归搜索所有子文件夹，并将子文件夹路径扁平化作为文件名的一部分。
    """

    # --- 1. 准备目标文件夹 ---
    dest_dir = Path(destination_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"目标文件夹: {dest_dir}")

    total_files_copied = 0

    # --- 2. 遍历所有源文件夹 ---
    for prefix, source_folder in source_map.items():
        source_path = Path(source_folder)

        # 检查源文件夹是否存在
        if not source_path.is_dir():
            print(f"警告: 找不到源文件夹 {source_path}, 已跳过 (前缀: '{prefix}')。")
            continue

        if recursive:
            print(f"处理: {source_path} (前缀: '{prefix}') [递归模式]")
            # .rglob('*') 会递归查找所有文件
            iterator = source_path.rglob("*")
        else:
            print(f"处理: {source_path} (前缀: '{prefix}') [非递归模式]")
            # .iterdir() 只遍历当前目录
            iterator = source_path.iterdir()

        # --- 3. 遍历并复制文件 ---
        for file_path in iterator:
            # 确保我们只处理文件
            if file_path.is_file():

                if recursive:
                    # 获取 "subdir/image.png"
                    relative_path = file_path.relative_to(source_path)
                    # 转换 "subdir/image.png" -> "subdir_image.png"
                    flat_name_part = str(relative_path).replace(os.path.sep, "_")
                    new_filename = f"{prefix}_{flat_name_part}"
                else:
                    # 非递归模式，直接获取文件名
                    # "image.png" -> "gen1_image.png"
                    new_filename = f"{prefix}_{file_path.name}"

                # 构造最终目标路径
                dest_file = dest_dir / new_filename

                # 复制文件
                shutil.copy(file_path, dest_file)
                total_files_copied += 1

    print(f"\n复制完成! 总共复制了 {total_files_copied} 个文件。")
    return total_files_copied


import random

import numpy as np

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:27:42.788589Z","iopub.execute_input":"2025-10-28T14:27:42.788924Z","iopub.status.idle":"2025-10-28T14:27:45.817803Z","shell.execute_reply.started":"2025-10-28T14:27:42.788886Z","shell.execute_reply":"2025-10-28T14:27:45.816809Z"},"jupyter":{"outputs_hidden":false}}
# 固定随机种子
import torch


def set_seed(seed):
    # 固定 Python 的随机种子
    random.seed(seed)
    # 固定 NumPy 的随机种子
    np.random.seed(seed)
    # 固定 PyTorch 的随机种子（CPU）
    torch.manual_seed(seed)
    # 如果使用 GPU，则固定其随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU，设置所有 GPU 的种子
    # 保证一些操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 调用函数固定种子
set_seed(fixed_seed)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:27:45.819631Z","iopub.execute_input":"2025-10-28T14:27:45.820009Z","iopub.status.idle":"2025-10-28T14:28:02.778074Z","shell.execute_reply.started":"2025-10-28T14:27:45.81998Z","shell.execute_reply":"2025-10-28T14:28:02.777133Z"},"jupyter":{"outputs_hidden":false}}
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm

# 数据转换
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

source_path = "/kaggle/input/efficientnet-data/data"
my_label_data = "/kaggle/input/efficientnet-data/my_label_data"

# ===== 关键修改：分离真实数据和生成数据 =====
real_data_path = "./real_data"  # 只包含真实数据
train_data_path = "./train_data"  # 训练数据（真实数据 + GAN生成数据）
val_data_path = "./val_data"  # 验证数据（只有真实数据）

# 创建目录
for path in [real_data_path, train_data_path, val_data_path]:
    os.makedirs(f"{path}/0", exist_ok=True)
    os.makedirs(f"{path}/1", exist_ok=True)

print("=" * 60)
print("步骤1: 收集所有真实数据（不包括GAN生成的图片）")
print("=" * 60)

# 复制所有真实标注的数据到 real_data_path
shutil.copytree(f"{my_label_data}/0", f"{real_data_path}/0", dirs_exist_ok=True)
shutil.copytree(f"{my_label_data}/1", f"{real_data_path}/1", dirs_exist_ok=True)

# 新标注的数据，分离状态的数据较多
shutil.copytree(
    "/kaggle/input/efficientnet-data/efficient_net_data_me/cropped_objects/0",
    f"{real_data_path}/0",
    dirs_exist_ok=True,
)
shutil.copytree(
    "/kaggle/input/efficientnet-data/efficient_net_data_me/cropped_objects/1",
    f"{real_data_path}/1",
    dirs_exist_ok=True,
)
shutil.copytree(
    "/kaggle/input/efficientnet-data/efficient2/cropped_objects/0",
    f"{real_data_path}/0",
    dirs_exist_ok=True,
)
shutil.copytree(
    "/kaggle/input/efficientnet-data/efficient2/cropped_objects/1",
    f"{real_data_path}/1",
    dirs_exist_ok=True,
)

real_count_0 = sum(
    1
    for file_name in os.listdir(f"{real_data_path}/0")
    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"))
)
real_count_1 = sum(
    1
    for file_name in os.listdir(f"{real_data_path}/1")
    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"))
)
print(
    f"真实数据统计: 类别0={real_count_0}, 类别1={real_count_1}, 总计={real_count_0 + real_count_1}"
)

print("\n" + "=" * 60)
print("步骤2: 划分真实数据为训练集和验证集 (80/20)")
print("=" * 60)

# 加载真实数据集
real_dataset = ImageFolder(root=real_data_path, transform=transform_val)
NUM_CLASSES = len(real_dataset.classes)

# 使用 sklearn 的 train_test_split 进行分层划分（Stratified Split）
from sklearn.model_selection import train_test_split

train_ratio = 0.8

# 获取所有样本的索引和标签
indices = list(range(len(real_dataset)))
labels = [label for _, label in real_dataset.samples]

# 使用 stratify 参数确保训练集和验证集都包含所有类别
train_indices, val_indices = train_test_split(
    indices,
    test_size=1 - train_ratio,
    stratify=labels,  # 关键参数：按标签分层划分
    random_state=fixed_seed,
)

# 打印每个类别的划分情况
from collections import Counter

train_labels = [labels[i] for i in train_indices]
val_labels = [labels[i] for i in val_indices]

train_label_counts = Counter(train_labels)
val_label_counts = Counter(val_labels)

for label in sorted(set(labels)):
    print(
        f"类别 {label}: 总数={labels.count(label)}, 训练={train_label_counts[label]}, 验证={val_label_counts[label]}"
    )

print(f"\n真实数据划分: 训练集={len(train_indices)}, 验证集={len(val_indices)}")

# 将训练集的真实数据复制到 train_data_path
for idx in train_indices:
    img_path, label = real_dataset.samples[idx]
    filename = os.path.basename(img_path)
    dest_path = f"{train_data_path}/{label}/{filename}"
    shutil.copy(img_path, dest_path)

# 将验证集的真实数据复制到 val_data_path
for idx in val_indices:
    img_path, label = real_dataset.samples[idx]
    filename = os.path.basename(img_path)
    dest_path = f"{val_data_path}/{label}/{filename}"
    shutil.copy(img_path, dest_path)

train_real_count_0 = len([f for f in os.listdir(f"{train_data_path}/0")])
train_real_count_1 = len([f for f in os.listdir(f"{train_data_path}/1")])
val_count_0 = len([f for f in os.listdir(f"{val_data_path}/0")])
val_count_1 = len([f for f in os.listdir(f"{val_data_path}/1")])

print(f"训练集真实数据: 类别0={train_real_count_0}, 类别1={train_real_count_1}")
print(f"验证集数据: 类别0={val_count_0}, 类别1={val_count_1}")

# 验证分层划分是否成功
if val_count_0 > 0 and val_count_1 > 0:
    print("✓ 验证集包含两个类别，分层划分成功！")
else:
    print("⚠️ 警告: 验证集缺少某个类别！")


print("\n" + "=" * 60)
if USE_GAN_DATA:
    print("步骤3: 向训练集添加GAN生成的数据（仅用于数据增强）")
else:
    print("步骤3: 跳过GAN数据，仅使用真实数据")
print("=" * 60)

if USE_GAN_DATA:
    # 定义GAN生成数据的源文件夹
    sources_to_copy = {
        "gen1": Path("/kaggle/input/efficientnet-data/generated_images_20251106_075700")
    }

    # 将GAN生成的图片添加到训练集的类别0
    copy_files_with_prefix(sources_to_copy, f"{train_data_path}/0")
else:
    print("已跳过添加GAN数据")

train_total_count_0 = len([f for f in os.listdir(f"{train_data_path}/0")])
train_total_count_1 = len([f for f in os.listdir(f"{train_data_path}/1")])

if USE_GAN_DATA:
    gan_count = train_total_count_0 - train_real_count_0
    print(f"添加了 {gan_count} 张GAN生成的图片到训练集")
    print(
        f"训练集最终统计: 类别0={train_total_count_0} (真实={train_real_count_0}, GAN={gan_count}), 类别1={train_total_count_1}"
    )
else:
    print(
        f"训练集统计: 类别0={train_total_count_0}, 类别1={train_total_count_1} (100% 真实数据)"
    )

print("\n" + "=" * 60)
print("步骤4: 创建DataLoader")
print("=" * 60)

# 创建数据集（训练集使用数据增强，验证集不使用）
train_dataset = ImageFolder(root=train_data_path, transform=transform_train)
val_dataset = ImageFolder(root=val_data_path, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)} (100% 真实数据)")
print("=" * 60)
print("✓ 数据准备完成！验证集只包含真实数据，可以可靠地评估模型性能")
print("=" * 60)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:02.780093Z","iopub.execute_input":"2025-10-28T14:28:02.780522Z","iopub.status.idle":"2025-10-28T14:28:03.367308Z","shell.execute_reply.started":"2025-10-28T14:28:02.780492Z","shell.execute_reply":"2025-10-28T14:28:03.366309Z"},"jupyter":{"outputs_hidden":false}}
import matplotlib.pyplot as plt
import torchvision


# 显示一批图像
def imshow(img, title=None):
    # 正确的反归一化（针对 ImageNet 统计数据）
    np_img = img.numpy()
    # 转置为 (H, W, C)
    np_img = np.transpose(np_img, (1, 2, 0))
    # ImageNet 的 mean 和 std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # 反归一化公式: img = img * std + mean
    np_img = std * np_img + mean
    np_img = np.clip(np_img, 0, 1)  # 将值限制在 [0, 1] 之间

    plt.figure(figsize=(10, 5))
    plt.imshow(np_img)
    if title:
        plt.title(title)
    plt.show()


# 从 train_loader 中获取一个批次的数据
data_iter = iter(train_loader)
images, labels = next(data_iter)  # 使用 next() 而不是 .next()

# 显示图像
imshow(
    torchvision.utils.make_grid(images[:8]),
    title=[f"Label: {label}" for label in labels[:8]],
)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:03.3684Z","iopub.execute_input":"2025-10-28T14:28:03.368639Z","iopub.status.idle":"2025-10-28T14:28:03.37244Z","shell.execute_reply.started":"2025-10-28T14:28:03.368615Z","shell.execute_reply":"2025-10-28T14:28:03.371593Z"},"jupyter":{"outputs_hidden":false}}
# torch.cuda.empty_cache()
# %pip install wandb
# import wandb
# wandb.login(key="152f9fe95a7ab860e0a400288743fa7139e84e5b")
# wandb.init(project='particle', tags=['efficient net nice'], name='efficient b1 net nice 32 batch size')

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:03.373549Z","iopub.execute_input":"2025-10-28T14:28:03.373902Z","iopub.status.idle":"2025-10-28T14:28:04.577886Z","shell.execute_reply.started":"2025-10-28T14:28:03.373875Z","shell.execute_reply":"2025-10-28T14:28:04.576905Z"},"jupyter":{"outputs_hidden":false}}
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision import models

# 检查是否有多个 GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 使用第一个 GPU
print(f"device: {device}")
# 加载 EfficientNet 模型
model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
# model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

# 将模型转移到 GPU 或 CPU
model = model.to(device)


# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:04.579191Z","iopub.execute_input":"2025-10-28T14:28:04.579858Z","iopub.status.idle":"2025-10-28T14:28:04.587481Z","shell.execute_reply.started":"2025-10-28T14:28:04.579825Z","shell.execute_reply":"2025-10-28T14:28:04.586543Z"},"jupyter":{"outputs_hidden":false}}
def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    epochs,
    save_path="/kaggle/working/model.pth",
):
    best_accuracy = 0.0  # 记录最佳精度
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}%"
        )

        # 检查当前精度是否高于最佳精度，如果是，则保存模型
        if epoch > 5 and accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with accuracy: {accuracy}%")


import torch

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:04.588955Z","iopub.execute_input":"2025-10-28T14:28:04.589403Z","iopub.status.idle":"2025-10-28T14:28:04.604872Z","shell.execute_reply.started":"2025-10-28T14:28:04.58933Z","shell.execute_reply":"2025-10-28T14:28:04.604015Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset


def train_k_fold(
    model,
    dataset,
    criterion,
    optimizer,
    epochs,
    k=5,
    batch_size=32,
    save_path="/kaggle/working/best_model.pth",
    stop_threshold=1e-6,
):
    kfold = KFold(n_splits=k, shuffle=True, random_state=fixed_seed)
    best_accuracy = 0.0  # 用于保存最佳模型的精度
    best_val_loss = float("inf")  # 用于保存最小验证损失

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k}")

        # 创建训练集和验证集
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            # 训练模式
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # 验证模式
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            accuracy = 100 * correct / total
            average_val_loss = val_loss / len(val_loader)  # 计算平均验证损失
            print(
                f"Epoch [{epoch+1}/{epochs}] Fold [{fold+1}/{k}], Train Loss: {running_loss/len(train_loader):.6f}, Validation Loss: {average_val_loss:.6f}, Accuracy: {accuracy:.2f}%"
            )

            # 保存最佳模型
            if accuracy >= best_accuracy:
                if average_val_loss < best_val_loss:
                    best_val_loss = average_val_loss
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), save_path)
                    print(
                        f"Best model saved with accuracy: {accuracy:.2f}%  Validation Loss: {average_val_loss:.6f}"
                    )
            # 提前停止条件
            if average_val_loss <= stop_threshold:
                print(
                    f"Stopping early: Validation loss {average_val_loss:.7f} reached threshold {stop_threshold}."
                )
                return

    print(
        f"K-Fold Training Completed. Best Accuracy: {best_accuracy:.2f}%  Best Validation Loss: {best_val_loss:.6f}"
    )


# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:04.605911Z","iopub.execute_input":"2025-10-28T14:28:04.606233Z"},"jupyter":{"outputs_hidden":false}}
# 定义损失函数和优化器

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练选项1: 使用简单的训练/验证划分（推荐，因为验证集已经正确分离）
train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    epochs=150,
    save_path="/kaggle/working/best_model.pth",
)

# 训练选项2: 使用K折交叉验证（需要使用只包含真实数据的数据集）
# 注意：K折交叉验证会在真实数据内部进行划分，不会用到GAN生成的数据
# 如果要使用K折，需要在真实数据上进行：
# real_dataset_for_kfold = ImageFolder(root=real_data_path, transform=transform_train)
# train_k_fold(
#     model=model,
#     dataset=real_dataset_for_kfold,  # 使用真实数据集进行K折
#     criterion=criterion,
#     optimizer=optimizer,
#     epochs=50,
#     k=5,
#     batch_size=64,
#     save_path="/kaggle/working/best_model.pth",
# )

# %% [code] {"jupyter":{"outputs_hidden":false}}
# 重新加载训练好的模型（仅在训练完成后运行此单元格）
import os

model_path = "/kaggle/working/best_model.pth"

if os.path.exists(model_path):
    print(f"加载已训练的模型: {model_path}")
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    # model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 将模型转移到 GPU 或 CPU
    model = model.to(device)
    print("✓ 模型加载成功")
else:
    print(f"⚠️ 模型文件不存在: {model_path}")
    print("请先运行训练代码，训练完成后会自动保存模型")
    print("如果已经训练完成，模型应该会在训练过程中自动使用（无需重新加载）")

import torch

# %% [code] {"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def evaluate(
    model,
    data_loader,
    device,
    class_names=["c", "s"],
    num_images=16,
    show_images=False,
    show_only_errors=False,
):
    """
    评估模型性能并可视化结果

    参数:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 计算设备 (cuda/cpu)
        class_names: 类别名称列表
        num_images: 要显示的图像数量
        show_images: 是否显示图像
        show_only_errors: 如果为True，只显示预测错误的样本；如果为False，显示所有样本
    """
    print(f"\n{'='*60}")
    print(f"开始评估 - 数据集大小: {len(data_loader.dataset)}")
    print(f"{'='*60}")

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 保存正类的概率分数，用于绘制PR曲线
    shown_images = 0  # 计数器，用于限制显示的图像数量
    model.eval()

    # 推理时间统计
    inference_times = []
    total_inference_time = 0.0
    batch_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            batch_count += 1
            # 先将数据传输到设备（不计入推理时间）
            images, labels = images.to(device), labels.to(device)

            # 记录推理开始时间
            if device.startswith('cuda'):
                torch.cuda.synchronize()  # 确保GPU操作完成
            batch_start_time = time.time()

            # 模型推理
            outputs = model(images)

            # 记录推理结束时间
            if device.startswith('cuda'):
                torch.cuda.synchronize()  # 确保GPU操作完成
            batch_end_time = time.time()

            # 计算本批次推理时间
            batch_inference_time = batch_end_time - batch_start_time
            inference_times.append(batch_inference_time)
            total_inference_time += batch_inference_time

            # 获取置信度
            confidences = torch.softmax(outputs, dim=1)  # 计算每个类别的置信度
            max_confidences, predicted = confidences.max(1)  # 获取每个图像的最高置信度

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 保存正类（类别1）的概率分数，用于PR曲线
            all_probs.extend(confidences[:, 1].cpu().numpy())

            # 转换图像回到原始的未归一化状态，用于显示
            images = images.cpu().numpy()
            images = images * np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
            images = images + np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
            images = np.clip(images, 0, 1)  # 将值限制在0-1之间

            if show_images and shown_images < num_images:
                if show_only_errors:
                    # 只显示预测错误的样本
                    incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]
                    if len(incorrect_indices) > 0:
                        num_to_show = min(
                            8, len(incorrect_indices), num_images - shown_images
                        )
                        indices_to_show = incorrect_indices[:num_to_show]
                    else:
                        indices_to_show = []
                else:
                    # 显示所有样本
                    num_to_show = min(8, len(images), num_images - shown_images)
                    indices_to_show = list(range(num_to_show))

                if len(indices_to_show) > 0:
                    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                    for i, idx in enumerate(indices_to_show):
                        ax = axes[i // 4, i % 4]
                        ax.imshow(np.transpose(images[idx], (1, 2, 0)))  # 调整维度顺序
                        # 显示预测类别、真实类别和置信度
                        ax.set_title(
                            f"Pred: {predicted[idx].item()} ({max_confidences[idx].item():.2f}), True: {labels[idx].item()}"
                        )
                        ax.axis("off")
                        shown_images += 1
                    # 隐藏未使用的子图
                    for i in range(len(indices_to_show), 8):
                        axes[i // 4, i % 4].axis("off")
                    plt.show()
    #                 wandb.log(
    #                     {
    #                         "Predictions": [
    #                             wandb.Image(fig, caption=f"Batch {shown_images // 8}")
    #                         ]
    #                     }
    #                 )

    cm = confusion_matrix(all_labels, all_preds)

    # 获取实际出现的类别
    unique_labels = sorted(set(all_labels) | set(all_preds))

    # 只使用实际出现的类别标签
    actual_class_names = [
        class_names[i] if i < len(class_names) else str(i) for i in unique_labels
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=actual_class_names
    )
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 检查是否只有一个类别
    if len(unique_labels) < 2:
        print(f"\n⚠️ 警告: 只检测到 {len(unique_labels)} 个类别")
        print(f"   实际标签: {set(all_labels)}")
        print(f"   预测标签: {set(all_preds)}")
        print("   无法计算精度-召回率曲线（需要至少2个类别）")
        print(f"Accuracy: {100 * correct / total}%")
        return

    # 计算精度和召回率（使用概率分数而不是预测标签）
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    average_precision = average_precision_score(all_labels, all_probs)

    # 绘制精度-召回率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, marker=".", label="Precision-Recall curve")
    plt.fill_between(recall, precision, alpha=0.1)
    plt.title(f"Precision-Recall Curve (AP = {average_precision:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.show()
    #     wandb.log({"Confusion Matrix": wandb.Image(fig)})

    # 计算精确率、召回率、F1分数
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # 打印详细的统计信息
    print(f"\n{'='*60}")
    print("评估结果总结")
    print(f"{'='*60}")
    print(f"总样本数: {total}")
    print(f"正确预测: {correct}")
    print(f"错误预测: {total - correct}")
    print(f"准确率 (Accuracy): {100 * correct / total:.2f}%")

    # 性能指标统计
    print(f"\n{'='*60}")
    print("性能指标 (Macro Average)")
    print(f"{'='*60}")
    print(f"精确率 (Precision): {precision_macro:.4f} ({precision_macro * 100:.2f}%)")
    print(f"召回率 (Recall): {recall_macro:.4f} ({recall_macro * 100:.2f}%)")
    print(f"F1分数 (F1-Score): {f1_macro:.4f} ({f1_macro * 100:.2f}%)")

    print(f"\n{'='*60}")
    print("性能指标 (Weighted Average)")
    print(f"{'='*60}")
    print(f"精确率 (Precision): {precision_weighted:.4f} ({precision_weighted * 100:.2f}%)")
    print(f"召回率 (Recall): {recall_weighted:.4f} ({recall_weighted * 100:.2f}%)")
    print(f"F1分数 (F1-Score): {f1_weighted:.4f} ({f1_weighted * 100:.2f}%)")

    # 推理时间统计
    print(f"\n{'='*60}")
    print("推理时间统计")
    print(f"{'='*60}")
    print(f"总批次数: {batch_count}")
    print(f"总推理时间: {total_inference_time:.4f} 秒")
    print(f"平均每批次推理时间: {total_inference_time / batch_count:.4f} 秒")
    print(f"平均每张图片推理时间: {total_inference_time / total * 1000:.2f} 毫秒")
    print(f"推理吞吐量: {total / total_inference_time:.2f} 张/秒")
    print(f"最快批次推理时间: {min(inference_times):.4f} 秒")
    print(f"最慢批次推理时间: {max(inference_times):.4f} 秒")

    # 按类别统计
    from collections import Counter

    label_counts = Counter(all_labels)
    pred_counts = Counter(all_preds)

    print("\n真实标签分布:")
    for label in sorted(label_counts.keys()):
        label_name = class_names[label] if label < len(class_names) else str(label)
        print(
            f"  {label_name}: {label_counts[label]} ({100*label_counts[label]/total:.1f}%)"
        )

    print("\n预测标签分布:")
    for label in sorted(pred_counts.keys()):
        label_name = class_names[label] if label < len(class_names) else str(label)
        print(
            f"  {label_name}: {pred_counts[label]} ({100*pred_counts[label]/total:.1f}%)"
        )
    print(f"{'='*60}\n")


evaluate(model, val_loader, device, show_images=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 加载测试集
test_dataset = ImageFolder(
    root="/kaggle/input/efficientnet-data/test", transform=transform
)

# 创建测试集 DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 使用相同的 evaluate 函数在测试集上评估
# evaluate(model, test_loader, device, show_images=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# 只显示预测错误的样本
evaluate(model, test_loader, device, show_images=True, show_only_errors=True)
