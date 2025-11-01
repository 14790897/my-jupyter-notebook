# SAGAN优化方案 - 从FID 80降到50以下

## 项目概述
- **架构**: SAGAN (Self-Attention GAN)
- **数据集**: 约300张粒子检测图片 (64x64灰度)
- **起始FID**: 80 (epoch 500)
- **目标FID**: < 50

## 当前实施的优化策略

### 1. 数据增强优化 (已优化)
**问题**: 过度增强会损害图像质量
**最终方案**:
- ✅ 随机水平翻转 (p=0.5)
- ✅ 随机垂直翻转 (p=0.5)
- ❌ ~~随机旋转~~ - 已移除，不适合粒子数据
- ❌ ~~随机仿射变换~~ - 已移除，不适合粒子数据
- ❌ ~~颜色抖动~~ - 已移除，会损害FID分数

**效果**: 最小化增强，保持图像质量，提高FID分数

### 2. Self-Attention机制
**问题**: 标准DCGAN难以捕捉长距离依赖关系
**解决方案**:
- 在Generator和Discriminator的16x16分辨率层添加Self-Attention
- 使用Query-Key-Value机制
- 带有可学习权重γ的残差连接

**效果**: 提升生成图像的全局一致性和细节质量

### 3. Spectral Normalization
**问题**: 小数据集上训练不稳定,Discriminator容易过强
**解决方案**:
- 在Discriminator的所有卷积层应用Spectral Normalization
- 限制判别器的Lipschitz常数

**效果**: 大幅提升训练稳定性,防止梯度爆炸/消失

### 4. 优化超参数

#### 训练参数 (当前配置)

- **Batch Size**: 16 (较大batch提供更好的梯度估计)
- **Learning Rate**: 0.0002 (DCGAN标准学习率)
- **Epochs**: 2000 (充分训练)
- **Latent Dim**: 128 (更丰富的潜在空间)
- **D/G训练比例**: 1:1 (平衡训练)

#### 优化器配置

- **Optimizer**: Adam
- **Beta1**: 0.0 (适合小数据集)
- **Beta2**: 0.9 (标准配置)
- **学习率调度**: StepLR (每500 epochs衰减0.5倍)

### 5. 训练稳定性增强

**新增优化**:

- ✅ **梯度裁剪** (max_norm=1.0) - 防止梯度爆炸
- ✅ **标签平滑** (real=0.9, fake=0.0) - 提高训练稳定性
- ✅ **学习率调度** (StepLR) - 动态调整学习速率

**效果**: 显著提升训练稳定性，避免mode collapse

### 6. FID评估优化 (自适应策略)

**智能评估频率**:

- Epoch 0-200: 每10个epoch (早期频繁监控)
- Epoch 200-800: 每20个epoch (中期稳定监控)
- Epoch 800+: 每50个epoch (后期精细调整)
- FID样本数: 500张 (更准确的评估)

## 预期改进效果

| 指标 | 初始值 | 当前最佳 | 目标值 |
|------|--------|----------|--------|
| FID Score | 150-200 | 80 | < 50 |
| 训练稳定性 | 中等 | 高 | 高 |
| 生成多样性 | 低 | 中-高 | 高 |
| 细节质量 | 低 | 中 | 高 |

### FID改进路线图

- **Epoch 0-200**: FID 150-200 → 80-100
- **Epoch 200-500**: FID 80-100 → 60-80  
- **Epoch 500-1000**: FID 60-80 → 50-60 (第一次学习率衰减)
- **Epoch 1000-1500**: FID 50-60 → 45-50 (第二次学习率衰减)
- **Epoch 1500+**: FID < 50 ✅ (目标达成)

## 训练监控建议

### 关键指标

1. **D(x)**: 应保持在 0.5-0.8 之间 (判别器对真实图像的置信度)
2. **D(G(z))**: 应逐渐接近 0.5 (生成器质量提升)
3. **Loss平衡**: G_loss和D_loss应相对平衡
4. **FID趋势**: 应持续下降，无长期停滞

### 警告信号

- ⚠️ **D_loss → 0**: 判别器过强，考虑减少训练步数
- ⚠️ **G_loss爆炸**: 学习率可能过高
- ⚠️ **FID停滞100+ epochs**: 考虑调整学习率或early stopping
- ⚠️ **D(x) < 0.5**: 判别器性能下降

## 进一步优化选项

如果FID在1500 epochs后仍未达到50:

1. **延长训练** → 2500-3000 epochs
2. **调整学习率调度** → 更早或更频繁的衰减
3. **增加网络容量** → ngf=96, ndf=96
4. **添加多层Self-Attention** → 在8x8和32x32分辨率
5. **尝试StyleGAN2** → 见`stylegan2.py`文件
6. **数据质量检查** → 移除低质量样本

## 代码结构

### 核心模块

- **`SelfAttention`**: Query-Key-Value自注意力机制
- **`Generator`**: DCGAN + Self-Attention + BatchNorm
- **`Discriminator`**: Spectral Norm + Self-Attention + LeakyReLU

### 训练循环特性

- ✅ 自适应FID计算频率
- ✅ 梯度裁剪 (max_norm=1.0)
- ✅ 标签平滑 (real=0.9)
- ✅ 学习率调度 (StepLR)
- ✅ 最佳模型自动保存
- ✅ 详细训练日志

## 当前配置 (gan.py)

```python
# 关键参数
batch_size = 16
learning_rate = 0.0002
num_epochs = 2000
latent_dim = 128
d_steps = 1
g_steps = 1
real_label_smooth = 0.9
fake_label_smooth = 0.0

# 学习率调度
StepLR(step_size=500, gamma=0.5)
```

## 文件输出

- **模型权重**: `./dcgan_weights/`
  - `generator_best_fid.pth` - 最佳FID模型
  - `discriminator_best_fid.pth` - 对应判别器
  - `best_fid_info.txt` - 最佳FID信息
  
- **生成图像**: `./dcgan_images/`
  - `fake_samples_epoch_XXX.png` - 每10 epochs生成样本

- **FID计算**: `./real_images_64x64_for_fid/` - 真实图像参考集

---

**创建时间**: 2025-11-01  
**最后更新**: 2025-11-01  
**状态**: ✅ 实施完成，训练中  
**目标**: FID < 50
