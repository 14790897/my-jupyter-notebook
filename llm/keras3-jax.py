# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-08T15:22:36.254483Z","iopub.execute_input":"2026-03-08T15:22:36.254627Z","iopub.status.idle":"2026-03-08T15:22:38.899986Z","shell.execute_reply.started":"2026-03-08T15:22:36.254610Z","shell.execute_reply":"2026-03-08T15:22:38.899063Z"}}
# 1. 环境准备 (极简安装)
!pip install --upgrade keras keras-hub
from IPython.display import clear_output
clear_output()
print("Keras 3 环境安装完成！")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-08T15:25:15.522673Z","iopub.execute_input":"2026-03-08T15:25:15.523118Z","iopub.status.idle":"2026-03-08T15:25:17.299965Z","shell.execute_reply.started":"2026-03-08T15:25:15.523082Z","shell.execute_reply":"2026-03-08T15:25:17.298424Z"}}
import os
# 【极其重要】强制指定 JAX 为后端，并分配全部 TPU 显存
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax
import keras
import keras_hub

print(f"检测到的 TPU 核心数: {len(jax.devices())}")

def print_tpu_memory(tag=""):
    print(f"\n[显存] {tag}")
    for i, d in enumerate(jax.devices()):
        m = d.memory_stats()
        used = m["bytes_in_use"] / 1024**3
        limit = m["bytes_limit"] / 1024**3
        print(f"  core {i}: {used:.2f} GB / {limit:.2f} GB")

print_tpu_memory("初始化后")

# ---------------------------------------------------------
# 2. 定义 TPU 分布式网格与 SPMD 切片策略 (替代原代码的 xs.mark_sharding)
# ---------------------------------------------------------
# 定义 1x8 的设备网格
device_mesh = keras.distribution.DeviceMesh(
    shape=(1, 8),
    axis_names=["batch", "model"],
    devices=jax.devices(),
)

# 使用正则表达式定义切片规则 (LayoutMap)
# Qwen2.5 在 KerasHub 中的实际层名前缀为 "qwen2_causal_lm/..."
layout_map = keras.distribution.LayoutMap(device_mesh)
# Embedding 按词汇维度切分
layout_map[".*embedding.*embeddings"] = (None, "model")
# Attention QKV/Output kernel：(hidden, heads*head_dim) -> 按 model 轴切
layout_map[".*attention.*(query|key|value|output)_dense.*kernel"] = (None, "model")
# FFN gate/up 切列，down 切行
layout_map[".*feedforward.*gate.*kernel"] = (None, "model")
layout_map[".*feedforward.*up.*kernel"] = (None, "model")
layout_map[".*feedforward.*down.*kernel"] = ("model", None)

# 全局应用分布策略
model_parallel = keras.distribution.ModelParallel(
    device_mesh=device_mesh,
    layout_map=layout_map,
    batch_dim_name="batch",
)
keras.distribution.set_distribution(model_parallel)

# ---------------------------------------------------------
# 3. 数据处理 (替代原代码复杂的 Tokenizer 和 data_generator)
# ---------------------------------------------------------
# KerasHub 会自动处理 Tokenization、Padding 以及将 Pad 的 Loss 设为零（屏蔽计算）。
# 你只需要提供纯文本格式的数据即可！
raw_data = [
    "<|im_start|>user\n1+1等于几？<|im_end|>\n<|im_start|>assistant\n等于2。<|im_end|>",
    "<|im_start|>user\n讲个笑话<|im_end|>\n<|im_start|>assistant\n我不会讲笑话。<|im_end|>"
] * 50

# ---------------------------------------------------------
# 4. 加载模型并注入 LoRA (替代原代码的 PEFT 和 AutoModel)
# ---------------------------------------------------------
print("正在将模型加载并切分到 TPU...")

# 注意：KerasHub 更推荐直接使用 Kaggle Models 提供的 Keras 预设 (Preset)
# 如果你想加载本地 Hugging Face 权重，可以使用 "hf://..." 路径或先进行权重转换
# Qwen2.5-32B 在 TPU v5-8 (128GB HBM) 上超出内存（权重64G + AdamW优化器状态~128G）
# 改用 7B：权重~14G + 优化器~28G，LoRA 仅训练少量参数，内存充裕
MODEL_PRESET = "hf://Qwen/Qwen2.5-7B-Instruct"

model = keras_hub.models.CausalLM.from_preset(
    MODEL_PRESET,
    dtype="bfloat16"
)
print_tpu_memory("模型加载后")

# KerasHub 原生支持 LoRA，一行代码搞定！
# 它会自动将 rank 应用到 Attention 和 FFN 层，并冻结基础骨干网络
print("正在添加 LoRA 适配器...")
# rank=32 在 32B 模型上消耗过多；7B 上用 rank=8 已足够
model.backbone.enable_lora(rank=8)
print_tpu_memory("enable_lora 后")
model.summary()

# ---------------------------------------------------------
# 5. 编译与训练 (替代原代码繁琐的 for 循环和 xm.mark_step)
# ---------------------------------------------------------
# 配置优化器，Keras 内部的 JAX 后端会自动处理所有的并行梯度计算和同步
optimizer = keras.optimizers.AdamW(learning_rate=2e-4)

# CausalLM 预设会自动处理 Next-Token Prediction 的 Loss 计算逻辑
model.compile(
    optimizer=optimizer,
    steps_per_execution=8,  # 每 8 步同步一次，补偿 batch_size=1 的吞吐损失
)
print_tpu_memory("compile 后")

print("\n--- 开始训练 ---")
model.fit(
    x=raw_data,
    batch_size=1,   # 降低每步激活内存
    epochs=1,
)
print_tpu_memory("训练结束后")

# ---------------------------------------------------------
# 6. 保存 LoRA 权重
# ---------------------------------------------------------
save_path = "./qwen-tpu-lora.lora.h5"
model.backbone.save_lora_weights(save_path)
print(f"✅ LoRA 模型权重已保存至 {save_path}")

# %% [code] {"execution":{"iopub.status.busy":"2026-03-08T15:23:43.058692Z","iopub.status.idle":"2026-03-08T15:23:43.059095Z","shell.execute_reply.started":"2026-03-08T15:23:43.058817Z","shell.execute_reply":"2026-03-08T15:23:43.058828Z"}}
# 1. 准备你的测试问题 
# 注意：为了让模型知道这是对话，我们需要按照 Qwen 的 Chat 格式拼装 Prompt
prompt_text = "<|im_start|>user\n你好！请用一句话介绍一下你自己，然后讲一个关于程序员的冷笑话。<|im_end|>\n<|im_start|>assistant\n"

print("正在思考中...")

# 2. 调用 generate 进行生成
# max_length 控制最大输出长度（包含输入提示词的长度）
response = model.generate(prompt_text, max_length=256)

# 3. 打印结果
print("\n=== 模型回复 ===")
# 截取 assistant 后面的内容，去掉我们输入的 prompt 部分以便阅读
if "<|im_start|>assistant\n" in response:
    clean_response = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    print(clean_response)
else:
    print(response)