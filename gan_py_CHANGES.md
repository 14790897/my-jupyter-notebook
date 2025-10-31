# gan.py 改动总结

## 概述
已将 `gan.py` 文件从WGAN-GP混合架构改为**标准DCGAN实现**。

## 主要改动

### 1. 参数配置
**之前:**
```python
batch_size = 16
latent_dim = 128
n_critic = 5
EVAL_FREQ = 50
```

**现在:**
```python
batch_size = 128  # DCGAN推荐
latent_dim = 100  # DCGAN标准
nc = 1  # 通道数
ngf = 64  # 生成器特征数
ndf = 64  # 判别器特征数
num_epochs = 200
```

### 2. Generator (生成器)
**改动:**
- 使用标准DCGAN生成器架构
- 使用 `ngf` 和 `nc` 参数使代码更灵活
- 清晰的注释标注每层的输出尺寸
- 移除了ResNet残差块代码（已注释）

**架构:** `100维 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64`

### 3. Discriminator (判别器)
**改动:**
- 移除Spectral Normalization
- 使用标准BatchNorm2d（第一层除外）
- 使用 `ndf` 和 `nc` 参数
- 输出经过Sigmoid激活
- 清晰的注释标注每层的输出尺寸

**架构:** `64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1`

### 4. Loss Function (损失函数)
**之前:**
```python
# WGAN-GP loss with gradient penalty
def discriminator_loss(real_output, fake_output):
    return torch.mean(F.relu(1.0 - real_output)) + torch.mean(F.relu(1.0 + fake_output))

def generator_loss(fake_output, label=None):
    return -torch.mean(fake_output)

def gradient_penalty(...):
    # 复杂的梯度惩罚计算
```

**现在:**
```python
# Binary Cross Entropy Loss
criterion = nn.BCELoss()
```

### 5. 优化器
**之前:**
```python
G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
```

**现在:**
```python
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
```
（命名更规范，符合DCGAN tutorial标准）

### 6. 训练循环
**之前 (WGAN-GP):**
```python
# 使用n_critic策略
# 计算梯度惩罚
# Wasserstein距离
D_loss = -torch.mean(real_output) + torch.mean(fake_output) + lambda_gp * gp
G_loss = -torch.mean(fake_output)
```

**现在 (标准DCGAN):**
```python
# 标准GAN训练流程
# (1) 更新判别器: maximize log(D(x)) + log(1 - D(G(z)))
errD_real = criterion(output_real, real_label)
errD_fake = criterion(output_fake, fake_label)
errD = errD_real + errD_fake

# (2) 更新生成器: maximize log(D(G(z)))
errG = criterion(output, real_label)
```

### 7. 数据加载
**简化:**
- 移除了复杂的数据增强（RandomDiscreteRotation等）
- 使用标准transforms
- 移除FID评估相关的复杂数据准备代码

### 8. 输出和保存
**之前:**
```python
./t_weights/
./images/
./generated_images/
./real_images_64x64_for_fid/
```

**现在:**
```python
./dcgan_weights/  # 模型权重
./dcgan_images/   # 训练过程样本
./dcgan_generated/  # 最终生成图像
```

### 9. 删除的功能
- ❌ FID评估代码（`calculate_fid`函数）
- ❌ 梯度惩罚函数（`gradient_penalty`）
- ❌ WGAN-GP相关的loss计算
- ❌ n_critic训练策略
- ❌ 复杂的数据增强
- ❌ 理论FID下限计算

### 10. 训练输出
**改进:**
- 更详细的训练统计: `D(x)`, `D(G(z))` before和after
- 每50个iteration输出一次
- 清晰的epoch进度显示
- 每10个epoch保存样本
- 每50个epoch保存checkpoint

## 代码质量改进

1. **命名规范化:**
   - `D_optimizer` → `optimizerD`
   - `G_optimizer` → `optimizerG`
   - `D_loss_plot` → `D_losses`
   - `G_loss_plot` → `G_losses`

2. **注释改进:**
   - 每个网络层都有输出尺寸注释
   - 训练步骤有清晰的分节注释
   - DCGAN原则在代码中明确标注

3. **代码结构:**
   - 遵循PyTorch官方DCGAN tutorial的结构
   - 更清晰的训练循环逻辑
   - 移除了冗余代码

## DCGAN核心原则验证

✅ 用strided convolutions替代pooling  
✅ 使用batch normalization  
✅ 移除全连接层  
✅ Generator用ReLU (输出层Tanh)  
✅ Discriminator用LeakyReLU  
✅ Adam优化器 (lr=0.0002, beta1=0.5)  
✅ BCE loss  
✅ 权重初始化 Normal(0, 0.02)  

## 使用说明

1. **准备数据:** 图像放在 `./train/data/` 目录
2. **运行训练:** 执行整个脚本
3. **查看结果:** 
   - 训练样本: `./dcgan_images/`
   - 模型权重: `./dcgan_weights/`
   - 生成图像: `./dcgan_generated/`
4. **调整参数:** 修改配置部分的参数

## 参考
- DCGAN论文: Radford et al., 2015
- PyTorch DCGAN Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
