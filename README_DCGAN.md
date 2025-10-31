# DCGAN训练指南

## 简介

这是一个标准的DCGAN (Deep Convolutional Generative Adversarial Network) 实现，用于生成64×64的粒子图像。

## 快速开始

### 1. 准备数据

将您的训练图像放在以下目录结构中：
```
./train/
  └── data/
      ├── image1.png
      ├── image2.png
      └── ...
```

### 2. 运行训练

按顺序运行notebook中的所有单元格：

1. **导入库和设置参数** (前4个cell)
2. **准备数据** (数据加载和预处理)
3. **定义模型** (Generator和Discriminator)
4. **训练模型** (主训练循环)
5. **查看结果** (可视化和生成图像)

### 3. 监控训练

训练过程中，检查以下目录：

- `./dcgan_images/` - 训练过程中的生成样本
- `./dcgan_weights/` - 保存的模型权重

### 4. 生成新图像

训练完成后，使用保存的生成器生成新图像：

```python
generator.load_state_dict(torch.load('./dcgan_weights/generator_epoch_199.pth'))
generator.eval()
noise = torch.randn(1, 100, 1, 1, device=device)
fake_image = generator(noise)
```

## 关键参数

### 训练参数

- `batch_size = 128` - 批次大小
- `learning_rate = 0.0002` - 学习率
- `num_epochs = 200` - 训练轮数
- `latent_dim = 100` - 潜在空间维度

### 架构参数

- `image_size = 64` - 图像大小 (64×64)
- `nc = 1` - 图像通道数 (1=灰度图, 3=彩色图)
- `ngf = 64` - 生成器特征图基数
- `ndf = 64` - 判别器特征图基数

## 训练建议

### 如果训练不稳定：

1. **降低学习率**: 将 `learning_rate` 从 0.0002 降至 0.0001
2. **调整批次大小**: 尝试 64 或 256
3. **增加训练轮数**: 如果图像质量还在提升，继续训练
4. **检查数据**: 确保训练数据质量良好且数量充足

### 期望的训练行为：

- 判别器损失 (D_loss) 应该稳定在 0.5-1.5 之间
- 生成器损失 (G_loss) 应该稳定在 0.5-2.0 之间
- D(x) 应该接近 0.5-0.8（真实图像的判别结果）
- D(G(z)) 应该接近 0.3-0.5（生成图像的判别结果）

## 硬件要求

- **推荐**: NVIDIA GPU (CUDA支持)
- **最低**: 4GB GPU内存
- **理想**: 8GB或更多GPU内存

没有GPU也可以训练，但速度会很慢。

## 文件说明

### 输出文件夹

- `./dcgan_images/` - 每10个epoch保存一次的生成样本
- `./dcgan_weights/` - 每50个epoch保存一次的模型权重
- `./dcgan_generated/` - 训练完成后的最终生成图像

### 模型文件

- `generator_epoch_X.pth` - 第X轮的生成器权重
- `discriminator_epoch_X.pth` - 第X轮的判别器权重

## DCGAN架构特点

本实现遵循原始DCGAN论文的所有建议：

✅ 使用strided convolutions代替pooling
✅ 使用batch normalization
✅ 移除全连接层
✅ 生成器使用ReLU激活（输出层Tanh）
✅ 判别器使用LeakyReLU激活
✅ 使用Adam优化器 (lr=0.0002, beta1=0.5)

## 常见问题

### Q: 生成的图像都很模糊？
A: 继续训练更多epoch，或检查训练数据质量。

### Q: 训练时loss出现NaN？
A: 降低学习率，或检查数据归一化是否正确。

### Q: 生成器和判别器loss差距很大？
A: 这可能表示训练失衡，尝试调整学习率或训练比例。

### Q: 如何改为生成彩色图像？
A: 将 `nc = 1` 改为 `nc = 3`，并调整数据加载部分移除灰度转换。

## 参考资料

- 原始DCGAN论文: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- PyTorch DCGAN教程: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## License

MIT License
