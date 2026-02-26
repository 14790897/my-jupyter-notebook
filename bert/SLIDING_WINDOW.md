# 使用滑动窗口优化长文本处理

## 改进概述

使用 `return_overflowing_tokens=True` 和 `stride` 参数，让 tokenizer 自动处理长文本的滑动窗口，无需手动字符级裁剪。

## 新方法的优势

### 1. 基于 Token 的滑动窗口（而非字符）

**旧方法（字符级裁剪）：**
```python
# 手动字符级裁剪，可能在词的中间切断
marker_pos = text.find(target_marker)
start_pos = max(0, marker_pos - 800)  # 字符位置
end_pos = min(len(text), marker_pos + 800)
cropped_text = text[start_pos:end_pos]  # 可能切断单词
```

**新方法（Token级滑动窗口）：**
```python
# tokenizer自动按token边界分割，保持完整性
inputs = tokenizer(
    text,
    max_length=512,              # 每个窗口最多512 tokens
    stride=128,                   # 窗口之间重叠128 tokens
    return_overflowing_tokens=True,  # 返回所有窗口
    truncation=True
)
# 自动找到包含mask的窗口
```

### 2. 自动处理长文本

**处理流程示例：**

```
原始文本（2000 tokens）:
[                    文本开始...          <mask>          ...文本结束                    ]

tokenizer自动分割成多个窗口：
窗口1: [文本开始...部分1]                           (512 tokens)
窗口2:      [部分1...部分2]                         (512 tokens, 与窗口1重叠128)
窗口3:           [部分2...<mask>...部分3]           (512 tokens, 包含mask ✓)
窗口4:                [部分3...部分4]               (512 tokens)
窗口5:                     [部分4...文本结束]       (512 tokens)

选择窗口3进行评分（因为它包含mask）
```

### 3. 代码对比

**旧实现（手动处理）：**
```python
# 需要手动检查文本长度
if len(text) > context_size * 2:
    # 手动裁剪
    start_pos = max(0, marker_pos - context_size)
    end_pos = min(len(text), marker_pos + context_size)
    text = text[start_pos:end_pos]
    
# 然后tokenizer截断
inputs = tokenizer(text, truncation=True, max_length=512)
```

**新实现（自动处理）：**
```python
# 一步到位，tokenizer自动处理所有情况
inputs = tokenizer(
    text,
    max_length=512,
    stride=128,
    return_overflowing_tokens=True,
    truncation=True
)

# 自动找到包含mask的chunk
for idx, chunk in enumerate(inputs["input_ids"]):
    if mask_token_id in chunk:
        selected_chunk = chunk
        break
```

## 技术细节

### 滑动窗口参数

```python
stride: int = 128  # 滑动步长
```

- **stride** 控制窗口之间的重叠
- stride=128 表示每个窗口与下一个窗口重叠 128 个 tokens
- 这确保了 mask token 不会落在窗口边界附近

### 示意图

```
max_length=512, stride=128:

Window 1: [0                                                    512]
Window 2:             [384                                     896]
Window 3:                         [768                        1280]
                                   ↑
                            mask token在这里
                            
重叠区域 = 512 - 128 = 384 tokens
```

## 性能影响

### 短文本（< 512 tokens）
- **旧方法**: 不裁剪，直接truncation
- **新方法**: 只返回一个窗口
- **性能**: 几乎无差异

### 中等文本（512-1024 tokens）
- **旧方法**: 字符级裁剪到1600字符
- **新方法**: 2-3个窗口，自动选择包含mask的窗口
- **性能**: 略慢（需处理多个窗口），但更精确

### 长文本（> 1024 tokens）
- **旧方法**: 字符级裁剪，可能切断单词
- **新方法**: 多个窗口，自动选择正确的窗口
- **性能**: 略慢，但准确性大幅提升

## 兼容性

### 向后兼容
```python
# 旧代码仍然可以工作
results = autoregressive_cloze_test(
    raw_text, 
    options_dict, 
    tokenizer, 
    model, 
    device,
    context_size=800  # 参数保留但不再使用
)
```

### API 无变化
- 所有函数签名保持不变
- `context_size` 参数保留（标记为已弃用）
- 返回值格式完全相同

## 使用示例

```python
from bert_utils import load_model, autoregressive_cloze_test

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 超长文本（自动处理滑动窗口）
long_text = """
[3000+ tokens的长文本...]
这里有 __1__ 个mask标记...
[更多文本...]
"""

options_dict = {
    1: ["option1", "option2", "option3", "option4"]
}

# 无需担心文本长度，自动处理！
results = autoregressive_cloze_test(
    long_text, 
    options_dict, 
    tokenizer, 
    model, 
    device
)
```

## 优势总结

✅ **更精确**: Token边界对齐，不会切断单词  
✅ **更智能**: 自动找到包含mask的窗口  
✅ **更简洁**: 无需手动管理文本裁剪  
✅ **更可靠**: 利用transformers库的内置功能  
✅ **更兼容**: 向后兼容，API无变化  

## 注意事项

1. **stride参数**: 默认128，可以根据需要调整
   - 更大的stride（如256）: 更快，但可能丢失上下文
   - 更小的stride（如64）: 更多重叠，上下文更完整

2. **padding**: 自动padding以处理不同长度的窗口

3. **内存使用**: 长文本会产生多个窗口，但只处理包含mask的窗口，所以影响很小

## 代码改动

### score_candidate_word 函数
- 添加 `stride` 参数
- 添加 `return_overflowing_tokens=True`
- 自动找到包含mask的chunk

### autoregressive_cloze_test 函数
- 移除手动裁剪逻辑
- 标记 `context_size` 为已弃用（保留以兼容）
- 简化代码逻辑

### 影响的文件
- `bert_utils.py`: 核心实现
- 所有测试代码自动受益，无需修改
