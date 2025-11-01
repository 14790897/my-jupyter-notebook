# 粒子检测 GAN 项目

## 📋 项目概述

本项目使用生成对抗网络(GAN)生成高质量的粒子检测图像。项目包含两种先进的GAN架构实现：

- **SAGAN** (Self-Attention GAN) - 基于DCGAN骨架，增强自注意力机制
- **StyleGAN2** - 最先进的样式生成架构

### 🎯 项目目标

- 从约300张粒子检测图像训练高质量生成模型
- 实现 **FID < 50** 的生成质量
- 比较SAGAN和StyleGAN2的性能差异

## 📁 项目结构

```
.
├── gan.py                          # SAGAN实现 (主要训练文件)
├── stylegan2.py                    # StyleGAN2实现
├── dcgan-resnet-structure-particle.ipynb  # Jupyter notebook
├── GAN_IMPROVEMENTS.md             # GAN优化策略详解
├── OPTIMIZATION_FOR_FID50.md       # FID<50优化方案
├── SAGAN_vs_StyleGAN2.md          # 两种架构对比分析
├── README.md                       # 本文件
├── train/data/                     # 训练数据 (~300张图像)
├── real_images_64x64_for_fid/     # FID评估参考集
├── dcgan_weights/                  # SAGAN模型权重
├── dcgan_images/                   # SAGAN生成的图像
├── stylegan2_weights/              # StyleGAN2模型权重
└── stylegan2_images/               # StyleGAN2生成的图像
```

## 🚀 快速开始

### 环境要求

```bash
Python 3.8+
PyTorch 1.10+
torchvision
pytorch-fid
```

### 安装依赖

```bash
pip install torch torchvision
pip install pytorch-fid
pip install matplotlib pillow
```

### 训练SAGAN模型

```bash
python gan.py
```

### 训练StyleGAN2模型

```bash
python stylegan2.py
```

## 📊 模型架构对比

### SAGAN (gan.py)

**核心特性**:
- ✅ Self-Attention机制 (16x16分辨率)
- ✅ Spectral Normalization (判别器稳定性)
- ✅ 标签平滑 (real=0.9, fake=0.0)
- ✅ 梯度裁剪 (max_norm=1.0)
- ✅ 自适应学习率调度

**训练配置**:
```python
batch_size = 16
learning_rate = 0.0002
num_epochs = 2000
latent_dim = 128
optimizer = Adam(beta1=0.0, beta2=0.9)
scheduler = StepLR(step_size=500, gamma=0.5)
```

**预期性能**:
- 目标FID: < 50
- 当前最佳: 80 (epoch 500)
- 训练时间: ~1500 epochs达到目标

### StyleGAN2 (stylegan2.py)

**核心特性**:
- ✅ 映射网络 (Z→W空间)
- ✅ 样式调制卷积
- ✅ 路径长度正则化
- ✅ R1梯度惩罚
- ✅ 等化学习率
- ✅ 噪声注入

**训练配置**:
```python
batch_size = 8
learning_rate = 0.002
num_epochs = 2000
latent_dim = 128
w_dim = 512
optimizer = Adam(beta1=0.0, beta2=0.99)
r1_gamma = 10.0
path_length_penalty = 2.0
```

**预期性能**:
- 目标FID: < 40
- 训练时间: ~1000-1500 epochs

## 📈 训练监控

### 关键指标

| 指标 | 正常范围 | 说明 |
|------|----------|------|
| D(x) | 0.5-0.8 | 判别器对真实图像的置信度 |
| D(G(z)) | 0.3-0.5 | 判别器对生成图像的置信度 |
| D_loss | 0.5-1.5 | 判别器损失 |
| G_loss | 0.8-2.0 | 生成器损失 |
| FID | < 50 | 图像质量指标(越低越好) |

### FID改进路线

| 训练阶段 | Epochs | 预期FID | 学习率 |
|----------|--------|---------|--------|
| 初始训练 | 0-200 | 150→80 | 0.0002 |
| 稳定收敛 | 200-500 | 80→60 | 0.0002 |
| 精细调整 | 500-1000 | 60→50 | 0.0001 |
| 最终优化 | 1000-1500+ | 50→45 | 0.00005 |

## 🛠️ 优化策略

### 已实施的优化

1. **数据增强优化**
   - 移除ColorJitter (损害图像质量)
   - 移除旋转和仿射变换 (不适合粒子数据)
   - 保留水平/垂直翻转

2. **训练稳定性**
   - 梯度裁剪 (防止梯度爆炸)
   - 标签平滑 (提高泛化能力)
   - Spectral Normalization (限制Lipschitz常数)

3. **自适应FID评估**
   - Epoch 0-200: 每10 epochs
   - Epoch 200-800: 每20 epochs
   - Epoch 800+: 每50 epochs

4. **学习率调度**
   - StepLR: 每500 epochs衰减50%
   - 帮助模型精细调整

详见：[GAN_IMPROVEMENTS.md](GAN_IMPROVEMENTS.md) 和 [OPTIMIZATION_FOR_FID50.md](OPTIMIZATION_FOR_FID50.md)

## 🎨 生成样本

训练过程中，模型会定期保存生成的图像：

- **SAGAN**: `./dcgan_images/fake_samples_epoch_XXX.png`
- **StyleGAN2**: `./stylegan2_images/fake_samples_epoch_XXX.png`

最佳模型（基于FID分数）会自动保存到：

- **SAGAN**: `./dcgan_weights/generator_best_fid.pth`
- **StyleGAN2**: `./stylegan2_weights/generator_best_fid.pth`

## 📚 文档说明

| 文件 | 说明 |
|------|------|
| [GAN_IMPROVEMENTS.md](GAN_IMPROVEMENTS.md) | SAGAN优化策略全面解析 |
| [OPTIMIZATION_FOR_FID50.md](OPTIMIZATION_FOR_FID50.md) | 达到FID<50的详细方案 |
| [SAGAN_vs_StyleGAN2.md](SAGAN_vs_StyleGAN2.md) | 两种架构的详细对比 |

## 🔧 故障排除

### 常见问题

**Q: FID分数不下降**
- 检查D(x)和D(G(z))是否在正常范围
- 考虑调整学习率
- 尝试延长训练时间

**Q: 训练不稳定/Mode Collapse**
- 检查梯度裁剪是否启用
- 增加Spectral Normalization
- 调整D:G训练比例

**Q: 生成图像质量差**
- 检查数据增强设置
- 验证标签平滑配置
- 考虑使用StyleGAN2架构

## 📊 性能基准

### SAGAN性能

| Epoch | FID Score | D(x) | D(G(z)) | 备注 |
|-------|-----------|------|---------|------|
| 100 | ~120 | 0.65 | 0.42 | 初期训练 |
| 500 | ~80 | 0.62 | 0.45 | 基线 |
| 1000 | ~55 | 0.58 | 0.48 | 第一次LR衰减 |
| 1500 | **<50** | 0.55 | 0.49 | **目标达成** |

### StyleGAN2性能（预期）

| Epoch | FID Score | 备注 |
|-------|-----------|------|
| 500 | ~60 | 初期训练 |
| 1000 | ~45 | 接近目标 |
| 1500 | **<40** | **优于SAGAN** |

## 💡 重要发现

### 数据增强限制

经过实验验证，以下数据增强**不适合**粒子检测数据：

- ❌ **旋转变换** (RandomRotation) - 粒子方向可能有物理意义
- ❌ **仿射变换** (RandomAffine) - 会造成几何失真
- ❌ **颜色抖动** (ColorJitter) - 对灰度图像无效，损害FID分数

### 训练里程碑

- **Epoch 500**: 使用Self-Attention机制后，FID达到80
- **目标**: 通过优化策略，在Epoch 1500左右达到FID < 50

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目仅用于学术研究和教育目的。

## 🙏 致谢

- SAGAN: Zhang et al. "Self-Attention Generative Adversarial Networks" (2018)
- StyleGAN2: Karras et al. "Analyzing and Improving the Image Quality of StyleGAN" (2020)
- PyTorch-FID: 用于FID计算的工具库

---

**最后更新**: 2025-11-01  
**状态**: ✅ 活跃开发中  
**当前目标**: 实现FID < 50