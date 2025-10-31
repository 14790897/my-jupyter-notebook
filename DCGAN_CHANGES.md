# DCGAN Notebook 改动总结

## 概述
已将混合架构的GAN notebook改为**标准DCGAN实现**，完全遵循Radford et al. 2015年的DCGAN论文。

## 主要改动

### 1. Generator (生成器)
**之前**: 
- 使用ResNet残差块结构
- 包含Upsample + Conv的复杂结构
- 适合高分辨率图像生成

**现在**: 
- 标准DCGAN生成器架构
- 纯粹使用ConvTranspose2d (fractional-strided convolutions)
- 架构流程: `latent(100) → 4×4 → 8×8 → 16×16 → 32×32 → 64×64`
- 使用BatchNorm2d和ReLU激活（输出层用Tanh）

### 2. Discriminator (判别器)
**之前**: 
- 使用Spectral Normalization
- 针对WGAN-GP优化
- 输出未经过Sigmoid

**现在**: 
- 标准DCGAN判别器架构
- 使用BatchNorm2d（第一层除外）
- 使用LeakyReLU(0.2)激活
- 输出层使用Sigmoid
- 架构流程: `64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1`

### 3. Loss Function (损失函数)
**之前**: 
- WGAN-GP loss
- Wasserstein距离
- 梯度惩罚 (Gradient Penalty)
- Hinge loss变体

**现在**: 
- Binary Cross Entropy (BCE) Loss
- 标准GAN损失函数
- 判别器: `max log(D(x)) + log(1 - D(G(z)))`
- 生成器: `max log(D(G(z)))`

### 4. Training Loop (训练循环)
**之前**: 
- 使用n_critic策略（每训练n次判别器，训练1次生成器）
- 计算梯度惩罚
- 复杂的损失计算

**现在**: 
- 标准DCGAN训练流程
- 每个batch同时更新D和G各一次
- 简洁的训练循环
- 详细的训练统计输出

### 5. 超参数调整
| 参数 | 之前 | 现在 | 说明 |
|------|------|------|------|
| batch_size | 16 | 128 | DCGAN论文推荐值 |
| learning_rate | 0.0002 | 0.0002 | 保持不变 |
| beta1 | 0.5 | 0.5 | Adam优化器beta1参数 |
| latent_dim | 128 | 100 | DCGAN标准维度 |
| num_epochs | 1500 | 200 | 简化训练 |
| ngf | - | 64 | 生成器特征图数量 |
| ndf | - | 64 | 判别器特征图数量 |

### 6. 权重初始化
保持DCGAN标准初始化策略：
- Conv层: Normal(0.0, 0.02)
- BatchNorm层: Normal(1.0, 0.02)

### 7. 文件结构改动
**之前**:
- `./t_weights/` - 保存模型权重
- `./images/` - 保存生成图像

**现在**:
- `./dcgan_weights/` - 保存DCGAN模型权重
- `./dcgan_images/` - 保存训练过程图像
- `./dcgan_generated/` - 保存最终生成图像

### 8. 删除的功能
- FID评估代码（可后续添加）
- WGAN-GP相关代码
- 梯度惩罚函数
- Spectral Normalization
- ResNet残差块

### 9. 新增功能
- 更清晰的训练统计输出
- 固定噪声可视化（用于观察训练进度）
- 详细的架构注释
- DCGAN原理说明文档

## DCGAN核心原则

本实现严格遵循以下DCGAN架构指导原则：

1. ✅ 用strided convolutions替代pooling层
2. ✅ 在G和D中都使用batch normalization
3. ✅ 移除全连接隐藏层
4. ✅ G中使用ReLU（输出层用Tanh）
5. ✅ D中使用LeakyReLU
6. ✅ 使用Adam优化器，lr=0.0002, beta1=0.5

## 使用建议

1. **数据准备**: 确保图像在 `./train/data/` 目录下
2. **训练**: 运行所有cell即可开始训练
3. **监控**: 查看 `./dcgan_images/` 目录中的中间结果
4. **评估**: 训练完成后查看 `./dcgan_generated/` 中的生成图像
5. **调优**: 如果训练不稳定，可以：
   - 降低学习率
   - 增加/减少batch size
   - 调整网络深度

## 参考文献

Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.
