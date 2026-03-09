# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-08T15:16:26.161363Z","iopub.execute_input":"2026-03-08T15:16:26.161548Z","iopub.status.idle":"2026-03-08T15:17:41.654955Z","shell.execute_reply.started":"2026-03-08T15:16:26.161529Z","shell.execute_reply":"2026-03-08T15:17:41.653703Z"}}
from IPython.display import clear_output

# 确保卸载可能冲突的库，并安装严格匹配的 TPU 稳定版环境
!pip uninstall -y torch torch_xla torchvision torchaudio
!pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install 'torch_xla[tpu]==2.8.0' -f https://storage.googleapis.com/libtpu-releases/index.html
!pip install -U peft transformers datasets accelerate

clear_output()
print("环境安装完成！")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-08T15:17:41.655572Z","iopub.execute_input":"2026-03-08T15:17:41.655750Z","iopub.status.idle":"2026-03-08T15:17:41.658980Z","shell.execute_reply.started":"2026-03-08T15:17:41.655730Z","shell.execute_reply":"2026-03-08T15:17:41.658223Z"}}
# 配置：设置模型路径
MODEL_NAME = "/kaggle/input/models/qwen-lm/qwen-3-5/transformers/qwen3.5-27b/1"  # 根据实际情况修改

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-08T15:17:41.659461Z","iopub.execute_input":"2026-03-08T15:17:41.659624Z","iopub.status.idle":"2026-03-08T15:17:52.817618Z","shell.execute_reply.started":"2026-03-08T15:17:41.659609Z","shell.execute_reply":"2026-03-08T15:17:52.816614Z"}}
import os
# 清除可能导致 XLA 多机识别错误的 Kaggle 环境变量
os.environ.pop('TPU_PROCESS_ADDRESSES', None)
os.environ.pop('CLOUD_TPU_TASK_ID', None)

import re
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

# 修复：torch.utils.checkpoint 内部会调用 getattr(torch, device_type)
# XLA 设备的 device_type 是 "xla"，需要将 torch_xla 注册到 torch.xla
torch.xla = torch_xla

# 1. 极其重要：激活 SPMD 单程序多数据模式
xr.use_spmd()
strkey2id = {
    "dp": 0,    # 数据并行
    "fsdp": 1,  # 全分片数据并行
    "mp": 2     # 模型/张量并行
}
# 2. 内嵌 SPMD 切片规则
# mesh_shape=(1,1,8): mp轴=8块卡
# PyTorch Linear 权重形状为 (out_features, in_features)
# 列并行 (Column Parallel) 切分 out_features 就是切第 0 维 -> ("mp", "fsdp")
# 行并行 (Row Parallel)    切分 in_features  就是切第 1 维 -> ("fsdp", "mp")
QWEN35_RULES = (
    # Embedding 沿 vocab 维度切分
    ("embed_tokens", ("mp", "fsdp")),

    # --- Attention 层 ---
    # Q/K/V 列并行：切分 out_features (第 0 维)
    ("(q_proj|k_proj|v_proj)$", ("mp", "fsdp")),
    # O 行并行：切分 in_features (第 1 维)
    ("o_proj$", ("fsdp", "mp")),

    # --- FFN (SwiGLU) 层 ---
    # Gate 和 Up 必须完全一致：都是列并行
    ("gate_proj$", ("mp", "fsdp")),
    ("up_proj$",   ("mp", "fsdp")),  # 修复原代码致命错误
    # Down 行并行
    ("down_proj$", ("fsdp", "mp")),

    # 输出层，沿 vocab 维度切分
    ("lm_head", ("mp", "fsdp")),
)

def find_rule(model):
    """根据模型配置返回对应的切片规则"""
    config_name = model.config.__class__.__name__
    if "Qwen3_5" in config_name or "Qwen2" in config_name:
        return QWEN35_RULES
    # 如果不是 Qwen，返回默认的 Qwen 规则（避免未定义错误）
    print(f"警告：模型类型 {config_name} 不是标准 Qwen 结构，使用默认规则")
    return QWEN35_RULES

# 2. 统一的切片函数：增加命中检查和日志输出
def partition_model(model, mesh, device):
    """将模型权重分片到多个 TPU 核心上"""
    rule = [(k, tuple([strkey2id[x] for x in v])) for k, v in find_rule(model)]
    sharded_count = 0
    
    # 首先将所有模块移动到 XLA 设备（包括 buffers）
    print("正在将所有模块和缓冲区移动到 XLA 设备...")
    for name, module in model.named_modules():
        module.to(device)
    
    # 然后对需要切片的层进行 SPMD 标记
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.Linear)):
            for rule_pattern, spec in rule:
                if re.findall(rule_pattern, name):
                    # 对权重进行 SPMD 切片标记
                    xs.mark_sharding(module.weight, mesh, spec)
                    sharded_count += 1
                    break
    
    print(f"✅ 成功切片了 {sharded_count} 个权重矩阵（含 language_model 路径）")
    if sharded_count == 0:
        raise Exception("切片失败！请检查层名称是否匹配。")
    return sharded_count

# 3. 初始化设备与超参数
DEVICE = xm.xla_device()
MAX_LENGTH = 512   # 从 1024 降低，减少 HBM 占用
BATCH_SIZE = 1     # 从 8 降低，全局 batch size

print(f"TPU 设备已初始化: {DEVICE}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-08T15:17:52.818186Z","iopub.execute_input":"2026-03-08T15:17:52.818543Z","iopub.status.idle":"2026-03-08T15:17:54.640325Z","shell.execute_reply.started":"2026-03-08T15:17:52.818524Z","shell.execute_reply":"2026-03-08T15:17:54.639406Z"}}
# 4. 加载 Tokenizer 与数据处理
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 模拟因果语言建模 (Causal LM) 的训练数据
raw_data = [
    {"messages": [{"role": "user", "content": "1+1等于几？"}, {"role": "assistant", "content": "等于2。"}]},
    {"messages": [{"role": "user", "content": "讲个笑话"}, {"role": "assistant", "content": "我不会讲笑话。"}]}
] * 50 # 复制扩大一点数据量以供测试

def process_data(examples):
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
    encodings = tokenizer(texts, padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
    
    # 构建 Causal LM 的 labels：将 pad 部分的 label 设为 -100，使其不参与 Loss 计算
    labels = encodings["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    encodings["labels"] = labels
    return encodings

dataset = Dataset.from_dict({"messages": [item["messages"] for item in raw_data]})
dataset = dataset.map(process_data, batched=True, batch_size=10, remove_columns=["messages"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 构建数据生成器
def data_generator(dataset, batch_size, mesh):
    idx = 0
    while True:
        batch_indices = range(idx, min(idx + batch_size, len(dataset)))
        if not batch_indices:
            idx = 0
            continue
        
        batch = dataset[batch_indices]
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        # 只沿 batch 维度（fsdp=index 1）切分，不切 seq 维度
        # mesh 轴: ('dp'=0, 'fsdp'=1, 'mp'=2)，shape=(1, num_devices, 1)
        xs.mark_sharding(input_ids, mesh, (1, None))
        xs.mark_sharding(attention_mask, mesh, (1, None))
        xs.mark_sharding(labels, mesh, (1, None))
        
        idx += batch_size
        yield input_ids, attention_mask, labels

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-08T15:17:54.641115Z","iopub.execute_input":"2026-03-08T15:17:54.641316Z","iopub.status.idle":"2026-03-08T15:19:56.435995Z","shell.execute_reply.started":"2026-03-08T15:17:54.641298Z","shell.execute_reply":"2026-03-08T15:19:56.434520Z"}}
# 4. 加载模型与 SPMD 切片
print("正在加载 Qwen-3.5 27B 基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False, 
    trust_remote_code=True
)

# 【配置修复】如果外壳缺失 vocab_size，从内部 language_model 复制过来
if not hasattr(base_model.config, "vocab_size"):
    if hasattr(base_model, "language_model"):
        print("检测到复合架构，正在同步 vocab_size...")
        base_model.config.vocab_size = base_model.language_model.config.vocab_size
    else:
        # 如果找不到，手动设为 Qwen 默认的 152064
        print("警告：无法自动获取 vocab_size，使用默认值 152064")
        base_model.config.vocab_size = 152064

base_model.config.pad_token_id = tokenizer.pad_token_id
# 「极其重要」没有这一行，梯度检查点会被默默跳过
base_model.enable_input_require_grads()base_model.gradient_checkpointing_enable()  # 节省激活值显存（torch.xla 已 monkey-patch）

# 定义 TPU Mesh
num_devices = xr.global_runtime_device_count()
# mesh_shape = (1, num_devices, 1)
mesh_shape = (1, 1, num_devices) # (dp=1, fsdp=1, mp=8)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))

print(f"正在将 27B 基础模型张量切片分布到 {num_devices} 个 TPU 核心...")
# 【核心逻辑】切片函数会先将所有模块移到 XLA 设备，然后对权重进行 SPMD 切片
partition_model(base_model, mesh, DEVICE)

# 最后再注入 LoRA 包装器
print("正在添加 LoRA 适配器...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-08T15:19:56.436703Z","iopub.execute_input":"2026-03-08T15:19:56.437258Z","iopub.status.idle":"2026-03-08T15:20:24.017512Z","shell.execute_reply.started":"2026-03-08T15:19:56.437237Z","shell.execute_reply":"2026-03-08T15:20:24.015725Z"}}
# 5. 原生 PyTorch + XLA 训练循环
model.train()
OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=2e-4)

# 确保优化器状态为 float32 (TPU 精度要求)
for state in OPTIMIZER.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.dtype is not torch.float32:
            state[k] = v.to(dtype=torch.float32)  # 修复：应该是 state[k] 而不是 state[v]

STEPS_PER_EPOCH = len(dataset) // BATCH_SIZE
train_gen = data_generator(dataset, BATCH_SIZE, mesh)

print("\n--- 开始训练 ---")
st = time.time()

for step in range(STEPS_PER_EPOCH):
    OPTIMIZER.zero_grad()
    
    input_ids, attention_mask, labels = next(train_gen)
    
    # Causal LM 自带 Loss 计算
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    OPTIMIZER.step()
    
    # 核心：触发 TPU XLA 图编译与执行
    xm.mark_step()
    
    if (step + 1) % 5 == 0 or step == 0:
        elapsed = time.time() - st
        print(f"Step {step+1}/{STEPS_PER_EPOCH} | Loss: {loss.item():.4f} | 耗时: {elapsed:.2f}s")

# 6. 保存 LoRA 权重
save_path = "./qwen3-27b-tpu-lora"
print(f"\n训练完成，正在保存至 {save_path} ...")
model.cpu()  # 保存前移回 CPU 内存
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 模型已保存至 {save_path}")