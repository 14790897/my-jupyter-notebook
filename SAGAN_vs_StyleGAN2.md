# GAN架构对比：SAGAN vs StyleGAN2

## 文件说明
- `gan.py` - SAGAN (Self-Attention GAN) 实现
- `stylegan2.py` - StyleGAN2 实现

## 架构对比

### SAGAN (Self-Attention GAN)
**优势：**
- ✅ Self-Attention机制捕捉长距离依赖
- ✅ Spectral Normalization稳定训练
- ✅ 相对简单，训练较快
- ✅ BatchNorm提供良好的归一化

**架构特点：**
```
Generator:
  - ConvTranspose2d layers (DCGAN backbone)
  - Self-Attention at 16x16 resolution
  - BatchNorm + ReLU
  
Discriminator:
  - Conv2d with Spectral Norm
  - Self-Attention at 16x16 resolution
  - LeakyReLU

Loss: Binary Cross-Entropy (BCE)
```

**训练参数：**
- Batch size: 16
- Learning rate: 0.0002
- Optimizer: Adam (beta1=0.0, beta2=0.9)
- Epochs: 2000

**预期FID：**
- 目标: < 50
- 当前最佳: 80 (epoch 500)
- 改进后预期: 50-60

---

### StyleGAN2
**优势：**
- ✅ **样式调制 (Style Modulation)** - 更好的特征解耦
- ✅ **映射网络 (Mapping Network)** - Z → W空间转换
- ✅ **路径长度正则化** - 更平滑的潜在空间
- ✅ **R1梯度惩罚** - 更强的判别器正则化
- ✅ **无渐进式增长伪影** - 改进的StyleGAN
- ✅ **等化学习率** - 所有层学习速度一致

**架构特点：**
```
Generator:
  - Mapping Network: Z (128) → W (512)
  - Constant Input (4x4)
  - Modulated Convolution blocks
  - Noise Injection for stochastic variation
  - Progressive synthesis: 4→8→16→32→64
  - AdaIN (Adaptive Instance Normalization)
  
Discriminator:
  - Equalized Convolution
  - Residual connections
  - Average pooling (no stride)
  - R1 regularization

Loss: Non-saturating logistic loss
```

**训练参数：**
- Batch size: 8 (更小，更精细)
- Learning rate G: 0.002 (更高)
- Learning rate D: 0.002
- Optimizer: Adam (beta1=0.0, beta2=0.99)
- R1 gamma: 10.0
- Path length penalty: 2.0
- Lazy regularization: 每16步

**预期FID：**
- 目标: < 40
- StyleGAN2通常比SAGAN低10-20分

---

## 关键差异

### 1. 生成器架构

| 特性 | SAGAN | StyleGAN2 |
|------|-------|-----------|
| 基础 | DCGAN (ConvTranspose2d) | 样式调制卷积 |
| 输入 | 噪声向量直接输入 | Constant + 样式注入 |
| 归一化 | BatchNorm | AdaIN |
| 注意力 | Self-Attention (1层) | 通过样式实现 |
| 潜在空间 | Z空间 | Z→W映射 |

### 2. 判别器架构

| 特性 | SAGAN | StyleGAN2 |
|------|-------|-----------|
| 归一化 | Spectral Norm | Equalized LR |
| 注意力 | Self-Attention | 无 |
| 正则化 | 无 | R1梯度惩罚 |
| 结构 | 标准卷积 | 残差连接 |

### 3. 损失函数

| 方面 | SAGAN | StyleGAN2 |
|------|-------|-----------|
| 基础损失 | BCE | Non-saturating logistic |
| 判别器正则化 | 无 | R1 penalty |
| 生成器正则化 | 无 | Path length penalty |
| 标签平滑 | 是 (0.9) | 否 |

### 4. 训练策略

| 方面 | SAGAN | StyleGAN2 |
|------|-------|-----------|
| 更新频率 | 1:1 (D:G) | 1:1 |
| 正则化频率 | 每步 | 延迟 (每16步) |
| 梯度裁剪 | 是 (max_norm=1.0) | 否 |
| 学习率调度 | StepLR (500步衰减) | CosineAnnealing |

---

## 性能预期

### FID分数预期
```
Epoch Range    SAGAN       StyleGAN2
0-100         150-200      120-150
100-200       80-120       70-100
200-500       60-80        50-70
500-1000      50-60        40-50
1000+         45-55        35-45
```

### 训练速度
- **SAGAN**: 更快 (~1.2x 基准)
  - 更简单的架构
  - 更大的batch size
  
- **StyleGAN2**: 较慢 (~0.8x 基准)
  - 更复杂的计算
  - 映射网络开销
  - 正则化计算

### 内存使用
- **SAGAN**: 中等
  - Batch size 16
  - Self-Attention需要额外内存
  
- **StyleGAN2**: 较高
  - Batch size 8 (但单样本占用更多)
  - 映射网络和样式调制
  - 梯度计算更复杂

---

## 使用建议

### 选择SAGAN (`gan.py`) 当：
- ✅ 需要更快的训练
- ✅ 计算资源有限
- ✅ 数据集较小 (<1000张)
- ✅ 需要简单易调试的架构
- ✅ FID 50-60 就足够

### 选择StyleGAN2 (`stylegan2.py`) 当：
- ✅ 追求最佳图像质量
- ✅ 需要更好的潜在空间控制
- ✅ 目标FID < 40
- ✅ 有足够的训练时间
- ✅ 需要样式迁移能力
- ✅ 数据集较大 (>1000张)

---

## 运行方式

### SAGAN
```bash
python gan.py
```

### StyleGAN2
```bash
python stylegan2.py
```

### 并行对比训练
```bash
# Terminal 1
python gan.py

# Terminal 2  
python stylegan2.py
```

---

## 输出目录

### SAGAN
```
./dcgan_weights/          # 模型权重
./dcgan_images/           # 训练过程图像
./dcgan_generated/        # 最终生成图像
```

### StyleGAN2
```
./stylegan2_weights/      # 模型权重
./stylegan2_images/       # 训练过程图像
./stylegan2_generated/    # 最终生成图像
```

---

## 调优建议

### SAGAN调优
1. 如果D太强: 减少`d_steps`或增加`g_steps`
2. 如果训练不稳定: 增加标签平滑值
3. 如果FID卡住: 调整学习率衰减时机
4. 如果模式崩溃: 增加正则化或减小学习率

### StyleGAN2调优
1. 如果D太强: 减小`r1_gamma`
2. 如果G训练慢: 减小`path_length_penalty`
3. 如果内存不足: 减小`batch_size`到4
4. 如果需要更快训练: 减少`lazy_regularization`频率

---

## 预期改进路径

### SAGAN改进轨迹
```
当前: FID 80 @ epoch 500
↓
优化数据增强: FID 70 @ epoch 500
↓
学习率调度: FID 60 @ epoch 800
↓
最终目标: FID 50 @ epoch 1500
```

### StyleGAN2预期轨迹
```
初始: FID 120 @ epoch 100
↓
快速改进: FID 70 @ epoch 300
↓
稳定优化: FID 50 @ epoch 700
↓
最终目标: FID 40 @ epoch 1500
```

---

## 总结

| 指标 | SAGAN | StyleGAN2 | 胜者 |
|------|-------|-----------|------|
| 图像质量 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | StyleGAN2 |
| 训练速度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | SAGAN |
| 内存效率 | ⭐⭐⭐⭐ | ⭐⭐⭐ | SAGAN |
| 稳定性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | StyleGAN2 |
| 可控性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | StyleGAN2 |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | SAGAN |

**推荐：**
- 快速原型/资源有限 → **SAGAN**
- 最佳质量/发表论文 → **StyleGAN2**
- 两者都试试，比较结果！
