# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:27:42.764355Z","iopub.execute_input":"2025-10-28T14:27:42.764569Z","iopub.status.idle":"2025-10-28T14:27:42.776652Z","shell.execute_reply.started":"2025-10-28T14:27:42.764544Z","shell.execute_reply":"2025-10-28T14:27:42.775942Z"}}
import shutil

from PIL import Image, ImageOps


def process_images_in_directory(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    # 遍历源目录中的文件
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)

        # 检查文件是否是图片
        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
        ):
            # 打开图片
            img = Image.open(file_path)

            # 保存原始图片到目标目录
            # original_save_path = os.path.join(target_dir, f'original_{file_name}')
            # img.save(original_save_path)

            # 上下翻转
            flipped_up_down = ImageOps.flip(img)
            flipped_up_down_save_path = os.path.join(
                target_dir, f"flipped_ud_{file_name}"
            )
            flipped_up_down.save(flipped_up_down_save_path)

            # 左右翻转
            flipped_left_right = ImageOps.mirror(img)
            flipped_left_right_save_path = os.path.join(
                target_dir, f"flipped_lr_{file_name}"
            )
            flipped_left_right.save(flipped_left_right_save_path)

            # 90度旋转
            rotated_90 = img.rotate(90)  # 逆时针旋转90度
            rotated_90_save_path = os.path.join(target_dir, f"rotated_90_{file_name}")
            rotated_90.save(rotated_90_save_path)

            # 270度旋转
            rotated_270 = img.rotate(270)  # 逆时针旋转270度
            rotated_270_save_path = os.path.join(target_dir, f"rotated_270_{file_name}")
            rotated_270.save(rotated_270_save_path)


# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:27:42.778281Z","iopub.execute_input":"2025-10-28T14:27:42.778552Z","iopub.status.idle":"2025-10-28T14:27:42.787551Z","shell.execute_reply.started":"2025-10-28T14:27:42.778526Z","shell.execute_reply":"2025-10-28T14:27:42.786731Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:27:42.788589Z","iopub.execute_input":"2025-10-28T14:27:42.788924Z","iopub.status.idle":"2025-10-28T14:27:45.817803Z","shell.execute_reply.started":"2025-10-28T14:27:42.788886Z","shell.execute_reply":"2025-10-28T14:27:45.816809Z"}}
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
set_seed(12)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:27:45.819631Z","iopub.execute_input":"2025-10-28T14:27:45.820009Z","iopub.status.idle":"2025-10-28T14:28:02.778074Z","shell.execute_reply.started":"2025-10-28T14:27:45.81998Z","shell.execute_reply":"2025-10-28T14:28:02.777133Z"}}
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm

# 数据转换
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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
print(f"真实数据统计: 类别0={real_count_0}, 类别1={real_count_1}, 总计={real_count_0 + real_count_1}")

print("\n" + "=" * 60)
print("步骤2: 划分真实数据为训练集和验证集 (80/20)")
print("=" * 60)

# 加载真实数据集
real_dataset = ImageFolder(root=real_data_path, transform=transform_val)
NUM_CLASSES = len(real_dataset.classes)

# 划分训练集和验证集（只使用真实数据）
train_ratio = 0.8
train_size = int(train_ratio * len(real_dataset))
val_size = len(real_dataset) - train_size

train_indices, val_indices = torch.utils.data.random_split(
    range(len(real_dataset)), [train_size, val_size]
)

print(f"真实数据划分: 训练集={train_size}, 验证集={val_size}")

# 将训练集的真实数据复制到 train_data_path
for idx in train_indices.indices:
    img_path, label = real_dataset.samples[idx]
    filename = os.path.basename(img_path)
    dest_path = f"{train_data_path}/{label}/{filename}"
    shutil.copy(img_path, dest_path)

# 将验证集的真实数据复制到 val_data_path
for idx in val_indices.indices:
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

print("\n" + "=" * 60)
print("步骤3: 向训练集添加GAN生成的数据（仅用于数据增强）")
print("=" * 60)

# 定义GAN生成数据的源文件夹
sources_to_copy = {
    "gen1": Path("/kaggle/input/efficientnet-data/generated_images_20251104_015533")
}

# 将GAN生成的图片添加到训练集的类别0
copy_files_with_prefix(sources_to_copy, f"{train_data_path}/0")

train_total_count_0 = len([f for f in os.listdir(f"{train_data_path}/0")])
train_total_count_1 = len([f for f in os.listdir(f"{train_data_path}/1")])
gan_count = train_total_count_0 - train_real_count_0

print(f"添加了 {gan_count} 张GAN生成的图片到训练集")
print(f"训练集最终统计: 类别0={train_total_count_0} (真实={train_real_count_0}, GAN={gan_count}), 类别1={train_total_count_1}")

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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:02.780093Z","iopub.execute_input":"2025-10-28T14:28:02.780522Z","iopub.status.idle":"2025-10-28T14:28:03.367308Z","shell.execute_reply.started":"2025-10-28T14:28:02.780492Z","shell.execute_reply":"2025-10-28T14:28:03.366309Z"}}
import matplotlib.pyplot as plt
import torchvision


# 显示一批图像
def imshow(img, title=None):
    img = img / 2 + 0.5  # 反归一化
    np_img = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # 将 (C, H, W) 转为 (H, W, C)
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


# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:03.3684Z","iopub.execute_input":"2025-10-28T14:28:03.368639Z","iopub.status.idle":"2025-10-28T14:28:03.37244Z","shell.execute_reply.started":"2025-10-28T14:28:03.368615Z","shell.execute_reply":"2025-10-28T14:28:03.371593Z"}}
# torch.cuda.empty_cache()
# %pip install wandb
# import wandb
# wandb.login(key="152f9fe95a7ab860e0a400288743fa7139e84e5b")
# wandb.init(project='particle', tags=['efficient net nice'], name='efficient b1 net nice 32 batch size')

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:03.373549Z","iopub.execute_input":"2025-10-28T14:28:03.373902Z","iopub.status.idle":"2025-10-28T14:28:04.577886Z","shell.execute_reply.started":"2025-10-28T14:28:03.373875Z","shell.execute_reply":"2025-10-28T14:28:04.576905Z"}}
from torchvision import models

# 检查是否有多个 GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 使用第一个 GPU
print(f"device: {device}")
# 加载 EfficientNet 模型
# model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

# 将模型转移到 GPU 或 CPU
model = model.to(device)


# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:04.579191Z","iopub.execute_input":"2025-10-28T14:28:04.579858Z","iopub.status.idle":"2025-10-28T14:28:04.587481Z","shell.execute_reply.started":"2025-10-28T14:28:04.579825Z","shell.execute_reply":"2025-10-28T14:28:04.586543Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:04.588955Z","iopub.execute_input":"2025-10-28T14:28:04.589403Z","iopub.status.idle":"2025-10-28T14:28:04.604872Z","shell.execute_reply.started":"2025-10-28T14:28:04.58933Z","shell.execute_reply":"2025-10-28T14:28:04.604015Z"}}
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
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
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


# %% [code] {"execution":{"iopub.status.busy":"2025-10-28T14:28:04.605911Z","iopub.execute_input":"2025-10-28T14:28:04.606233Z"}}
# 定义损失函数和优化器

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练选项1: 使用简单的训练/验证划分（推荐，因为验证集已经正确分离）
train_model(model, criterion, optimizer, train_loader, val_loader, epochs=50)

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

# %% [code]
# 重新加载 EfficientNet 模型
# model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# 将模型转移到 GPU 或 CPU
model = model.to(device)

import torch

# %% [code]
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


def evaluate(
    model, data_loader, device, class_names=["c", "s"], num_images=16, show_images=False, show_only_errors=False
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
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    shown_images = 0  # 计数器，用于限制显示的图像数量
    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # 获取置信度
            confidences = torch.softmax(outputs, dim=1)  # 计算每个类别的置信度
            max_confidences, predicted = confidences.max(1)  # 获取每个图像的最高置信度

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
                        num_to_show = min(8, len(incorrect_indices), num_images - shown_images)
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
                        axes[i // 4, i % 4].axis('off')
                    plt.show()
                    if shown_images >= num_images:
                        break
    #                 wandb.log(
    #                     {
    #                         "Predictions": [
    #                             wandb.Image(fig, caption=f"Batch {shown_images // 8}")
    #                         ]
    #                     }
    #                 )

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    # 计算精度和召回率
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    average_precision = average_precision_score(all_labels, all_preds)

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
    print(f"Accuracy: {100 * correct / total}%")


evaluate(model, val_loader, device, show_images=True)


# %% [code]
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
evaluate(model, test_loader, device, show_images=True)

# %% [code]
# 只显示预测错误的样本
evaluate(model, test_loader, device, show_images=True, show_only_errors=True)
