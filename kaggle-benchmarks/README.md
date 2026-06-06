# Kaggle Benchmarks — 使用说明

## 文件说明

| 文件 | 说明 |
|------|------|
| `complex_bench.py` | 3 个任务的复杂 benchmark（主脚本） |
| `example_task.py` | 简单示例任务 |
| `kaggle_benchmarks_reference.md` | SDK 参考文档 |
| `github_issue_cn_region_block.md` | CN 地区限制 bug report |

---

## 环境准备

### 1. 安装依赖
```bash
pip install kaggle-benchmarks python-dotenv
```

### 2. 每次运行前刷新 API Key（有效期 1 小时）
```bash
echo y | kaggle benchmarks auth
# 会自动写入 .env 文件
```

---

## 本地运行

```bash
# 直接运行，-u 参数实时显示输出
python -u complex_bench.py
```

预计耗时：约 6 分钟 / 约 $0.20
- Task 1 (The Inheritance Puzzle)：~80s
- Task 2 (Code Bug Hunt)：~200s
- Task 3 (Constrained Creativity)：~80s

---

## 推送到 Kaggle 远程执行

> ⚠️ 注意：目前远程执行会因 CN 地区限制报错，详见 `github_issue_cn_region_block.md`

```bash
# 推送前先刷新 Key
echo y | kaggle benchmarks auth

# Task 1: The Inheritance Puzzle
kaggle b tasks push the-inheritance-puzzle -f complex_bench.py --wait

# Task 2: Code Bug Hunt
kaggle b tasks push code-bug-hunt -f complex_bench.py --wait

# Task 3: Constrained Creativity
kaggle b tasks push constrained-creativity -f complex_bench.py --wait
```

### 推送命令参数说明

| 参数 | 说明 |
|------|------|
| `the-inheritance-puzzle` | task slug（由 `@kbench.task(name=...)` 中的名称自动转换：空格→短横线，全小写） |
| `-f complex_bench.py` | 指定脚本文件 |
| `--wait` | 等待远程执行完成后返回结果（不加则只推送不等结果） |

---

## 本地运行结果（已验证，11/11 通过）

| Task | 名称 | 断言 | 延迟 | 费用 | 模型 |
|------|------|------|------|------|------|
| 1 | The Inheritance Puzzle | 4/4 ✅ | ~80s | $0.05 | gemini-3-flash-preview |
| 2 | Code Bug Hunt | 3/3 ✅ | ~200s | $0.11 | gemini-3-flash-preview |
| 3 | Constrained Creativity | 4/4 ✅ | ~80s | $0.05 | gemini-3-flash-preview |

---

## 常见问题

### `.env` 加载失败
`source .env` 在 Windows Git Bash 上可能因环境变量过大报错：
```
environment is too large for exec
```
**解决**：脚本顶部已有 `from dotenv import load_dotenv; load_dotenv()`，直接 `python -u complex_bench.py` 即可，无需手动 source。

### 断言匹配注意事项
LLM 输出格式有变体，断言已针对性调整：
- 数字带逗号：`445,000` 而非 `445000`
- 运算符带空格：`n + 1` 而非 `n+1`
- 措辞差异：`same list object` 而非 `shallow`
