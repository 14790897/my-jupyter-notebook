# SAGAN 损失函数切换指南

## 概述

SAGAN代码现在支持两种损失函数：
1. **BCE Loss** (Binary Cross-Entropy) - DCGAN传统方法
2. **Hinge Loss** - SAGAN原论文推荐方法

## 如何切换损失函数

### 步骤1: 修改配置参数

在代码的参数配置部分（约第68行），找到：

```python
# ============ LOSS FUNCTION CONFIGURATION ============
# Choose loss function: 'bce' or 'hinge'
loss_type = 'bce'  # Options: 'bce' (BCE Loss) or 'hinge' (Hinge Loss)
# =====================================================
```

修改 `loss_type` 的值：
- `loss_type = 'bce'` - 使用BCE损失（默认）
- `loss_type = 'hinge'` - 使用Hinge损失

### 步骤2: 运行训练

直接运行代码，系统会自动：
- 根据损失函数类型配置判别器输出层（BCE用Sigmoid，Hinge不用）
- 使用相应的损失函数进行训练
- 显示当前使用的损失函数信息

## 两种损失函数对比

### BCE Loss (Binary Cross-Entropy)

**数学公式**:
```
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
L_G = -E[log D(G(z))]
```

**特点**:
- ✅ 训练稳定，尤其适合小数据集
- ✅ 配合标签平滑效果好
- ✅ 判别器输出通过Sigmoid，范围为[0,1]
- ✅ DCGAN经典配置，经过大量验证
- ❌ 可能存在梯度消失问题
- ❌ 饱和区域梯度弱

**推荐场景**:
- 数据集较小（< 1000张）
- 首次训练/快速原型
- 需要稳定的训练过程
- 计算资源有限

**配置**:
```python
loss_type = 'bce'
real_label_smooth = 0.9  # 标签平滑
fake_label_smooth = 0.0
```

---

### Hinge Loss（合页损失）

**数学公式**:
```
L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]
L_G = -E[D(G(z))]
```

**特点**:
- ✅ 理论上更强的边界约束
- ✅ SAGAN原论文推荐
- ✅ 通常在大数据集上表现更好
- ✅ 不使用Sigmoid，判别器输出为原始logits
- ✅ 梯度更稳定，不易饱和
- ❌ 训练可能不如BCE稳定
- ❌ 在小数据集上可能过于激进
- ❌ 不使用标签平滑

**推荐场景**:
- 数据集较大（> 1000张）
- BCE训练后FID停滞不前
- 追求更好的图像质量
- 有充足的训练时间

**配置**:
```python
loss_type = 'hinge'
# 注意：Hinge Loss不使用标签平滑
```

---

## 实验建议

### 方案A: 稳妥起见（推荐）

1. **先用BCE Loss训练**:
   ```python
   loss_type = 'bce'
   num_epochs = 1500
   ```
   - 目标：达到FID < 60

2. **如果BCE效果好（FID < 55）**:
   - 继续用BCE训练到FID < 50
   
3. **如果BCE停滞（FID卡在60-70）**:
   - 切换到Hinge Loss尝试突破
   ```python
   loss_type = 'hinge'
   num_epochs = 2000
   ```

### 方案B: 并行对比

同时运行两个训练：
```bash
# Terminal 1 - BCE Loss
# 修改代码: loss_type = 'bce'
python SAGAN.py

# Terminal 2 - Hinge Loss  
# 修改代码: loss_type = 'hinge'
# 修改输出目录避免冲突
python SAGAN.py
```

然后比较两者的FID分数。

---

## 训练监控差异

### BCE Loss 监控指标

```
D(x): 应保持在 0.5-0.8 之间
D(G(z)): 应逐渐接近 0.5
Loss_D: 通常在 0.5-1.5 之间
Loss_G: 通常在 0.8-2.5 之间
```

### Hinge Loss 监控指标

```
D(x): 可能大于1（因为没有Sigmoid）
D(G(z)): 初始可能为负值，逐渐接近0
Loss_D: 通常在 0.3-1.2 之间（比BCE低）
Loss_G: 可能为负值，绝对值应逐渐减小
```

**警告**: Hinge Loss的数值范围与BCE不同，这是正常的！

---

## 超参数建议

### 使用BCE Loss时

```python
batch_size = 16
learning_rate = 0.0002
real_label_smooth = 0.9
fake_label_smooth = 0.0
```

### 使用Hinge Loss时

```python
batch_size = 16
learning_rate = 0.0002  # 或稍微降低到 0.0001
# 不使用标签平滑
```

如果Hinge Loss训练不稳定，尝试：
- 降低学习率: `learning_rate = 0.0001`
- 增加判别器更新频率: `d_steps = 2`

---

## 预期效果对比

### 小数据集（~300张图像）

| Epoch | BCE Loss FID | Hinge Loss FID | 说明 |
|-------|--------------|----------------|------|
| 100 | 120-150 | 130-160 | BCE更稳定 |
| 500 | 70-90 | 75-95 | 差距不大 |
| 1000 | 55-65 | 50-60 | Hinge可能更好 |
| 1500 | 50-55 | 45-50 | Hinge优势显现 |

**结论**: 在小数据集上，两者差异不大。BCE可能更容易达到初步目标。

---

## 故障排除

### Hinge Loss训练崩溃

**症状**: Loss爆炸，生成图像质量急剧下降

**解决方案**:
1. 降低学习率: `learning_rate = 0.0001`
2. 增加梯度裁剪: `max_norm = 0.5`
3. 切回BCE Loss

### Hinge Loss训练慢

**症状**: FID下降很慢

**解决方案**:
1. 增加训练epochs: `num_epochs = 2500`
2. 调整学习率调度: `step_size = 300`
3. 确保Spectral Normalization正常工作

### BCE Loss FID停滞

**症状**: FID卡在60-70不下降

**解决方案**:
1. 尝试切换到Hinge Loss
2. 调整标签平滑: `real_label_smooth = 0.95`
3. 增加模型容量: `ngf = 96, ndf = 96`

---

## 代码变更说明

### 修改的部分

1. **判别器架构** (`Discriminator` class):
   - 添加 `use_sigmoid` 参数
   - Sigmoid层变为可选

2. **损失函数** (新增):
   - `discriminator_hinge_loss()` 函数
   - `generator_hinge_loss()` 函数

3. **训练循环**:
   - 根据 `loss_type` 使用不同的损失计算
   - BCE和Hinge分支清晰分离

4. **配置参数**:
   - 新增 `loss_type` 配置选项

### 保持不变的部分

- 生成器架构
- Self-Attention机制
- Spectral Normalization
- 学习率调度
- FID评估逻辑
- 所有其他优化策略

---

## 快速参考

### 使用BCE Loss（默认）

```python
# 配置参数部分
loss_type = 'bce'
real_label_smooth = 0.9
fake_label_smooth = 0.0
```

### 切换到Hinge Loss

```python
# 配置参数部分
loss_type = 'hinge'
# 标签平滑参数会被忽略
```

### 验证配置

运行代码后，检查输出：
```
Starting Optimized SAGAN (Self-Attention GAN) Training Loop...
Loss Function: BCE (Binary Cross-Entropy)
  - Label Smoothing: Real=0.9, Fake=0.0
  - Discriminator Output: Sigmoid activated (0-1)
```

或

```
Starting Optimized SAGAN (Self-Attention GAN) Training Loop...
Loss Function: HINGE (Hinge Loss)
  - No label smoothing (Hinge loss doesn't use it)
  - Discriminator Output: Raw logits
```

---

## 总结

- **BCE Loss**: 稳定可靠，适合快速验证和小数据集
- **Hinge Loss**: 潜力更大，适合追求极致质量
- **建议**: 先用BCE建立baseline，如有需要再尝试Hinge
- **切换简单**: 只需修改一个参数 `loss_type`

**实验是关键**！不同数据集可能有不同的最佳选择。

---

**创建时间**: 2025-11-01  
**适用版本**: SAGAN.py (带损失函数切换功能)  
**相关文档**: GAN_IMPROVEMENTS.md, OPTIMIZATION_FOR_FID50.md
