# 分层划分修复说明

## 问题诊断

### 原始问题
使用 `torch.utils.data.random_split()` 随机划分数据时，**没有考虑类别平衡**，导致验证集可能只包含一个类别。

```python
# ❌ 错误的方式 - 可能导致验证集只有一个类别
train_indices, val_indices = torch.utils.data.random_split(
    range(len(real_dataset)), [train_size, val_size]
)
```

### 为什么会出现这个问题？

1. **数据集结构**：ImageFolder 按文件夹顺序加载，类别0的所有样本在前，类别1的所有样本在后
2. **随机划分**：如果某个类别样本较少，随机划分可能完全跳过该类别
3. **结果**：验证集可能只包含类别0或类别1，导致混淆矩阵错误

## 解决方案：分层划分 (Stratified Split)

### 核心思路
对每个类别**分别**进行 80/20 划分，确保训练集和验证集都包含所有类别。

```python
# ✓ 正确的方式 - 分层划分
# 1. 按类别分组样本
class_indices = defaultdict(list)
for idx, (img_path, label) in enumerate(real_dataset.samples):
    class_indices[label].append(idx)

# 2. 对每个类别分别划分
train_indices = []
val_indices = []

for label, indices in class_indices.items():
    np.random.shuffle(indices)
    class_train_size = int(train_ratio * len(indices))
    
    train_indices.extend(indices[:class_train_size])
    val_indices.extend(indices[class_train_size:])
```

### 关键改进

1. **按类别分组**：使用 `defaultdict` 将每个类别的样本索引分别收集
2. **独立划分**：对每个类别分别应用 80/20 划分
3. **验证检查**：自动检查验证集是否包含所有类别

## 预期效果

### 修复前
```
类别 0: 总数=500, 训练=?, 验证=?
类别 1: 总数=300, 训练=?, 验证=?

验证集数据: 类别0=0, 类别1=160  ❌ 只有一个类别！
⚠️ ValueError: The number of FixedLocator locations (1) does not match the number of labels (2)
```

### 修复后
```
类别 0: 总数=500, 训练=400, 验证=100
类别 1: 总数=300, 训练=240, 验证=60

真实数据划分: 训练集=640, 验证集=160
训练集真实数据: 类别0=400, 类别1=240
验证集数据: 类别0=100, 类别1=60
✓ 验证集包含两个类别，分层划分成功！
```

## 优势

### 1. 类别平衡保证
- 每个类别都按相同比例（80/20）划分
- 验证集**必然**包含所有类别

### 2. 代表性更好
- 验证集能够代表真实数据分布
- 评估指标更可靠

### 3. 避免偏差
- 防止模型在缺少某类别的验证集上获得虚假高分
- 确保混淆矩阵、精度、召回率等指标有效

## 与 sklearn 的对比

如果使用 sklearn，可以这样做：

```python
from sklearn.model_selection import train_test_split

# 获取所有样本路径和标签
samples = [path for path, _ in real_dataset.samples]
labels = [label for _, label in real_dataset.samples]

# 分层划分
train_samples, val_samples, train_labels, val_labels = train_test_split(
    samples, labels, 
    test_size=0.2, 
    stratify=labels,  # 关键参数！
    random_state=42
)
```

但我们的手动实现更灵活，可以直接操作索引。

## 验证修复成功

运行代码后，检查输出：

1. ✓ 每个类别都显示训练/验证数量
2. ✓ 验证集包含两个类别
3. ✓ 显示 "验证集包含两个类别，分层划分成功！"
4. ✓ 混淆矩阵能够正常显示

## 文件位置

- 主代码：`c:\git-program\particle_detect\notebook\efficient_net\net.py`
- 修复说明：`c:\git-program\particle_detect\notebook\STRATIFIED_SPLIT_FIX.md`
