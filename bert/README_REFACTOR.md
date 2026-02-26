# BERT完形填空测试代码重构说明

## 重构概述

本次重构将重复的代码提取到公共函数库中，大幅简化了主测试文件的结构。

## 文件结构

### 新增文件

**`bert_utils.py`** - 公共函数库，包含以下功能：

#### 核心函数

1. **`setup_device()`** - 自动选择GPU或CPU设备
2. **`load_model(model_name, device)`** - 加载预训练MLM模型和分词器
3. **`score_sentence(sentence, ...)`** - 计算句子的整体loss值（旧方法，保留用于向后兼容）
4. **`score_candidate_word(text_with_mask, candidate_word, ...)`** - 使用mask token计算候选词的精确loss（推荐方法）
   - 对单个token的候选词：只计算mask位置的loss（更精确）
   - 对多个token的短语：计算整体loss
5. **`autoregressive_cloze_test(...)`** - 自回归完形填空答题的核心函数（已更新使用新评分方法）
6. **`simple_cloze_test(...)`** - 简单的单题完形填空测试
7. **`clean_text(text, target_marker, option)`** - 文本清理和替换
8. **`crop_context(text, marker, context_size)`** - 裁剪文本上下文
9. **`print_results_table(results, answers)`** - 打印答题结果表格

### 重构后的文件

**`cloze_test.py`** - 主测试文件，现在使用公共函数库

## 重构前后对比

### 重构前
每个测试代码块都包含约60-80行重复代码：
- 模型加载代码重复4次
- 评分循环代码重复4次
- 设备选择代码重复4次

### 重构后
每个测试代码块简化为约10-15行：
```python
from bert_utils import load_model, autoregressive_cloze_test

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 原始文本
raw_text = """..."""

# 选项字典
options_dict = {...}

# 执行自回归答题
results = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=1, end_idx=20
)
```

## 代码改进

### 1. 可维护性
- 公共逻辑集中管理，修改时只需改一处
- 清晰的函数划分，职责明确

### 2. 可读性
- 主测试文件更简洁，突出测试数据
- 函数名称直观，易于理解

### 3. 可复用性
- 公共函数可用于新的测试场景
- 支持自定义参数配置

### 4. 代码质量
- 添加完整的类型提示
- 添加详细的文档字符串
- 统一的代码风格

## 使用示例

### 简单测试（单题）
```python
from bert_utils import load_model, simple_cloze_test

tokenizer, model, device = load_model("roberta-base")

prompt_template = "Despite the {} evidence, the jury found it difficult to reach a unanimous verdict."
options = ["overwhelming", "vague", "insufficient", "unreliable"]

results = simple_cloze_test(prompt_template, options, tokenizer, model, device)
```

### 自回归测试（多题）
```python
from bert_utils import load_model, autoregressive_cloze_test

tokenizer, model, device = load_model("roberta-base")

raw_text = """..."""  # 包含 __1__, __2__ 等标记
options_dict = {
    1: ["option1", "option2", "option3", "option4"],
    2: ["option1", "option2", "option3", "option4"],
    # ...
}

results = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=1, end_idx=20
)
```

## 测试内容

重构后的代码包含以下测试：

1. **BERT 简单题目测试** - 单题零样本测试
2. **2022年考研英语一完形填空** (20题)
3. **2023年考研英语一完形填空** (20题)
4. **2019年上海英语高考完形填空** (15题)
5. **2019年上海英语春考完形填空** (15题，题号41-55)

## 依赖项

- `torch` - PyTorch深度学习框架
- `transformers` - Hugging Face Transformers库
- `re` - 正则表达式（标准库）
- `typing` - 类型提示（标准库）

## 注意事项

1. 代码中的一些linter警告（如"Module level import not at top of file"）是由于Jupyter notebook的单元格结构，这是正常的
2. `transformers`库的import警告可以忽略，这是运行时依赖
3. 确保在使用前已安装所有依赖项：`pip install torch transformers`

## 性能

重构不影响性能：
- 使用相同的算法和模型
- 相同的GPU加速支持
- 相同的动态上下文窗口优化

### 评分方法改进

新版本使用了更精确的评分方法：

**旧方法** (score_sentence):
- 将候选词填入句子，计算整个句子的loss
- 受到其他词的影响较大

**新方法** (score_candidate_word):
- 使用mask token替换目标位置
- 单词情况：只计算mask位置的loss，完全隔离其他词的影响（更精确）
- 短语情况：自动检测多词短语，使用整体loss评分
- 智能适配单词和短语两种情况

这使得模型能够更准确地评估每个候选词在特定位置的适配度。

## 后续改进建议

1. 添加结果缓存功能
2. 支持批量处理多个文本
3. 添加更多的评分策略
4. 支持其他预训练模型（BERT、ALBERT等）
5. 添加单元测试
