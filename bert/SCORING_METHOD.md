# 新评分方法使用示例

## 改进说明

新的 `score_candidate_word` 函数使用了更精确的评分方法：

### 工作原理

1. **单词情况**（如 "overwhelming", "vague"）：
   ```python
   # 使用 <mask> token，只计算该位置的loss
   text_with_mask = "Despite the <mask> evidence, ..."
   # 只计算 mask 位置预测 "overwhelming" 的loss
   loss = score_candidate_word(text_with_mask, "overwhelming", ...)
   ```
   
2. **短语情况**（如 "coped with", "hinted at"）：
   ```python
   # 自动检测多词短语，使用整体loss
   text_with_mask = "...that <mask> consciousness..."
   # 将 <mask> 替换为 "hinted at"，计算整体loss
   loss = score_candidate_word(text_with_mask, "hinted at", ...)
   ```

### 优势

**旧方法的问题**：
```python
# 旧方法：直接填入候选词，计算整个句子的loss
sentence = "Despite the overwhelming evidence, ..."
loss = score_sentence(sentence, ...)
# 问题：loss受整个句子的影响，不精确
```

**新方法的优势**：
```python
# 新方法：使用mask，只关注填空位置
text_with_mask = "Despite the <mask> evidence, ..."
loss = score_candidate_word(text_with_mask, "overwhelming", ...)
# 优势：只计算mask位置的loss，更精确
```

### 代码示例

#### 简单测试
```python
from bert_utils import load_model, score_candidate_word

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 准备测试句子（使用mask token）
text_with_mask = f"Despite the {tokenizer.mask_token} evidence, the jury found it difficult to reach a unanimous verdict."

# 测试各个候选词
options = ["overwhelming", "vague", "insufficient", "unreliable"]
results = {}

for opt in options:
    loss = score_candidate_word(text_with_mask, opt, tokenizer, model, device)
    results[opt] = loss
    print(f"{opt}: loss = {loss:.4f}")

# 找出最佳选项
best_option = min(results, key=results.get)
print(f"\n最佳选项: {best_option}")
```

#### 自回归测试（已集成）
```python
from bert_utils import load_model, autoregressive_cloze_test

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 原始文本（使用 __n__ 标记）
raw_text = """
The term was __1__ around the notion that some aspects of plant behavior 
could be __2__ to intelligence in animals.
"""

# 选项字典
options_dict = {
    1: ["coined", "discovered", "collected", "issued"],
    2: ["attributed", "directed", "compared", "confined"],
}

# 执行自回归答题（内部自动使用新的评分方法）
results = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=1, end_idx=2
)

# 结果会自动打印，也可以通过返回值访问
for i, result in results.items():
    print(f"题目 {i}: {result['letter']}. {result['option']} (loss: {result['loss']:.4f})")
```

### 技术细节

#### 单个token的精确评分
```python
# 1. 输入带有 <mask> 的句子
inputs = tokenizer(text_with_mask, return_tensors="pt")

# 2. 构造labels：只在mask位置计算loss
labels = torch.full_like(inputs["input_ids"], fill_value=-100)  # -100 表示忽略
mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()[0]
labels[0, mask_index] = candidate_id  # 只在mask位置设置目标ID

# 3. 计算loss（只针对mask位置）
outputs = model(**inputs, labels=labels)
loss = outputs.loss.item()  # 这是纯粹针对填空的loss
```

#### 多个token的降级处理
```python
# 对于 "coped with" 这样的短语：
# 1. 检测到有2个tokens: ["cop", "ed", "with"] 
# 2. 使用整体loss方法
text_filled = text_with_mask.replace("<mask>", "coped with")
inputs = tokenizer(text_filled, return_tensors="pt")
inputs["labels"] = inputs["input_ids"].clone()
outputs = model(**inputs)
loss = outputs.loss.item()
```

### 性能影响

- **准确性提升**：单词情况下更精确
- **速度影响**：几乎无影响（tokenization开销很小）
- **兼容性**：完全向后兼容，旧代码无需修改

### 适用场景

✅ **推荐使用新方法的场景**：
- 完形填空测试
- 词汇适配度评估
- 单词选择任务

⚠️ **可以使用旧方法的场景**：
- 需要评估整个句子的流畅度
- 不关心特定位置的精确度
- 向后兼容旧代码

### 注意事项

1. RoBERTa模型使用 `<mask>` token，可通过 `tokenizer.mask_token` 获取
2. BERT模型使用 `[MASK]` token  
3. 对于多词短语，自动降级到整体loss方法
4. 所有现有代码都已自动使用新方法，无需手动修改
