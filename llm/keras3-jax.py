# =============================================================================
# 【踩坑经验总结】Keras 3 + JAX + TPU v5e-8 微调大模型避坑指南
# =============================================================================
#
# ❌ 坑 1：模型尺寸必须匹配 TPU 显存
#   - Qwen2.5-32B bfloat16 权重 64GB + AdamW 优化器状态 ~128GB → OOM
#   - 换 7B：权重 14GB，LoRA 仅训练 ~50MB 参数，优化器状态极小，完全放得下
#
# ❌ 坑 2：不能用 dtype="float32" 加载模型
#   - KerasHub 的 from_preset 分两阶段：先 build（在 core 0 分配临时 buffer），
#     再加载权重并 sharding。build 阶段 LayoutMap 还未生效，所有 buffer 挤在
#     单卡 16GB 上，float32 在第 28 层 FFN build 时必然 OOM。
#   - 必须用 dtype="bfloat16"，bfloat16 的临时 buffer 刚好撑过 build 阶段。
#
# ❌ 坑 3：不能覆盖 CausalLM 的 loss 参数
#   - model.compile(loss=SparseCategoricalCrossentropy(...)) 会切断 KerasHub
#     内置的 padding mask 机制，导致模型去学 padding token，loss 卡在 0.49。
#   - 正确做法：compile 不传 loss，让 CausalLM 自动屏蔽 padding。
#
# ❌ 坑 4：LoRA 变量是 bfloat16，精度只有 ~0.78%
#   - enable_lora 新建的变量继承层自身的 bfloat16 dtype，
#     全局 set_dtype_policy("float32") 对已有层无效。
#   - 无法在 build 前切到 float32（见坑 2），只能用更大的学习率补偿：
#     2e-3 替代通常的 1e-4~5e-4。
#
# ❌ 坑 5：steps_per_execution=8 会掩盖训练是否真的在更新
#   - 每 8 步才 sync 一次，每个 epoch 只打印一次 loss，看不出 loss 是否在变。
#   - 调试期间先用 steps_per_execution=1，确认 loss 逐步下降后再改回 8。
#
# ❌ 坑 6：loss 每个 epoch 完全一致（如 0.4751/0.4751/0.4751）
#   - 原因组合：bfloat16 精度低 + 学习率太小 + 覆盖了内置 loss（坑 3+4）
#   - 修复：移除自定义 loss + 提高学习率到 2e-3
#
# ❌ 坑 7：keras.config.dtype_policy_scope() 是幽灵 API
#   - 在当前稳定版 Keras 3 中不存在，调用会报 AttributeError。
#
# ❌ 坑 8：自写 LayoutMap 规则时 LoRA 变量名写错
#   - LoRA 变量名是 lora_kernel_a / lora_kernel_b，不是 lora_a_kernel / lora_b_kernel。
#   - 写错名字 → 正则不匹配 → LoRA 变量取不到 sharding spec → 行为未定义。
#
# ❌ 坑 10：Adam/AdamW + bfloat16 LoRA 在 ModelParallel 下优化器状态被量化导致权重不更新
#   - AdamW 的 weight_decay 直接乘在 bfloat16 权重上，精度损失后实际更新量≈0
#   - Adam 的二阶动量（v_t）在 bfloat16 下也会被量化，导致 effective lr 接近 0
#   - 现象：每个 epoch 第 1 步 loss 完全一样，等于没有学习
#   - 修复：换用 SGD（官方 LoRA 示例做法），无动量累积，bfloat16 下更新量可靠
#   - TPU 上 lr 设 1e-2（GPU 官方用 1e-4，bfloat16 精度低需提高约 100x）
#
# ❌ 坑 11：QwenCausalLMPreprocessor 默认在每条样本前插入 BOS token
#   - add_start_token=True（默认），每条训练文本前会被插入 <|endoftext|>
#   - 训练输入：<|endoftext|><|im_start|>user\n...
#   - 推理输入：<|im_start|>user\n...（没有 BOS）
#   - 两者分布不同 → 模型推理时无法复现训练时见过的对话格式 → 看起来"不懂对话"
#   - 修复：model.preprocessor.add_start_token = False（ChatML 本身已有结构标记）
#
# ❌ 坑 9：DataParallel 在 7B 模型上必然 OOM，缩 sequence_length 也没用
#   - DataParallel 每核持完整权重 14.47GB + 反向传播激活值 ~6.5GB = 20.97GB > 15.75GB。
#   - 激活值主要来自 28 层 f32 注意力分数矩阵(112MB × 28 = 3.14GB)和 FFN 中间值。
#   - 哪怕把 sequence_length 降到 256，节省的激活内存也不够补足 5.2GB 的超支。
#   - 唯一出路是 ModelParallel，权重切分到 8 核：14.47/8 ≈ 1.8GB/核，激活值也随之切分。
#
# ✅ LoRA + ModelParallel 的正确做法（官方 gemma_backbone_test.py 验证）：
#   - lora_kernel_a / lora_kernel_b 全部 replicated (None, None)，不需要手动 shard
#   - 直接用 QwenBackbone.get_layout_map(device_mesh) 覆盖基础权重
#   - LoRA 不在 layout_map 中 → 自动 replicated → XLA 自动 slice/all-reduce，梯度正确
#
# ✅ 当前可用的最优配置（TPU v5e-8 / Qwen2.5-7B）：
#   - ModelParallel + QwenBackbone.get_layout_map（不用手写规则）
#   - dtype="bfloat16"（加载时必须）
#   - enable_lora(rank=16)
#   - SGD(learning_rate=1e-2)  ← 官方示例做法，不用 Adam/AdamW！
#   - compile() 不传 loss
#   - preprocessor.add_start_token = False  ← ChatML 有自己的起始标记，禁用 BOS
#   - preprocessor.sequence_length = 512  ← 避免截断长对话
#   - steps_per_execution=8（正式训练）
#   - batch_size=1（ModelParallel 无整除限制）
# =============================================================================
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-09T03:33:37.484226Z","iopub.execute_input":"2026-03-09T03:33:37.484434Z","iopub.status.idle":"2026-03-09T03:33:40.090050Z","shell.execute_reply.started":"2026-03-09T03:33:37.484415Z","shell.execute_reply":"2026-03-09T03:33:40.089035Z"}}
# 1. 环境准备 (极简安装)
!pip install --upgrade keras keras-hub
from IPython.display import clear_output
clear_output()
print("Keras 3 环境安装完成！")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-09T03:34:28.113723Z","iopub.execute_input":"2026-03-09T03:34:28.114008Z","iopub.status.idle":"2026-03-09T03:35:13.933502Z","shell.execute_reply.started":"2026-03-09T03:34:28.113987Z","shell.execute_reply":"2026-03-09T03:35:13.932195Z"}}
import os
# 【极其重要】强制指定 JAX 为后端，并分配全部 TPU 显存
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax
import keras
import keras_hub

# 不开全局混合精度，避免加载时权重膨胀到 float32 导致 OOM
# LoRA 局部升维到 float32 的逢魔在下方的 dtype_policy_scope 里做

print(f"检测到的 TPU 核心数: {len(jax.devices())}")

def print_tpu_memory(tag=""):
    print(f"\n[显存] {tag}")
    for i, d in enumerate(jax.devices()):
        m = d.memory_stats()
        used = m["bytes_in_use"] / 1024**3
        limit = m["bytes_limit"] / 1024**3
        print(f"  core {i}: {used:.2f} GB / {limit:.2f} GB")

print_tpu_memory("初始化后")

# %% [code] {"execution":{"iopub.status.busy":"2026-03-09T03:35:59.495066Z","iopub.execute_input":"2026-03-09T03:35:59.495704Z","iopub.status.idle":"2026-03-09T03:44:26.737792Z","shell.execute_reply.started":"2026-03-09T03:35:59.495682Z","shell.execute_reply":"2026-03-09T03:44:26.736320Z"}}
# ---------------------------------------------------------
# 2. 定义 TPU 分布策略
# ---------------------------------------------------------
# 使用 ModelParallel 将基础权重切分到 8 核，LoRA 矩阵自动保持 replicated。
#
# 【为什么不用自定义 LayoutMap 规则？】
# LoRA 变量名是 lora_kernel_a / lora_kernel_b（不是 lora_a_kernel）。
# 官方 Gemma/Llama 测试（gemma_backbone_test.py L126-156）确认：
#   lora_kernel_a → spec=(None, None, None)，lora_kernel_b → spec=(None, None)
#   即全部 replicated，这是 CORRECT 的。
# XLA 会自动处理：base_output（sharded）+ lora_output（replicated → 按需 slice）。
# 梯度也能正确 all-reduce，因为 JAX 对 replicated 变量自动做跨设备梯度求和。
#
# 【为什么不用 DataParallel？】
# DataParallel 每核持完整权重 14.47GB + 反向传播激活 ~6.5GB = 20.97GB > 15.75GB OOM。
# 无论怎么缩减 sequence_length 都救不回来。
device_mesh = keras.distribution.DeviceMesh(
    shape=(1, 8),
    axis_names=("batch", "model"),
    devices=jax.devices(),
)
# 使用官方内置的 layout_map，正确覆盖所有基础权重；LoRA 权重不在其中，自动 replicated
layout_map = keras_hub.models.QwenBackbone.get_layout_map(
    device_mesh, model_parallel_dim_name="model", data_parallel_dim_name="batch"
)
# ⚠️ Qwen2.5-7B 的 attention_output.kernel 形状是 (28, 128, 3584)，
# 官方默认规则 (model_dim, None, batch_dim) 会对 dim0=28 按 8 核切分，
# 但 28 % 8 ≠ 0 → ValueError。
# 改为对 dim2=3584（3584/8=448，整除）切分，保持 tensor parallel 效果。
del layout_map["transformer_layer.*attention_output.kernel"]
layout_map["transformer_layer.*attention_output.kernel"] = (None, None, "model")
model_parallel = keras.distribution.ModelParallel(
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
  "<|im_start|>user\n能讲个笑话吗？<|im_end|>\n<|im_start|>assistant\n<think>\n用户可能是傻福\n没错，用户是傻福，哈哈哈哈哈哈哈哈哈哈哈哈。\n</think>\n你是傻福吧，哈哈哈哈哈哈哈哈哈哈哈哈<|im_end|>\n",
  "<|im_start|>user\n大语言模型的 SFT 阶段主要目的是什么？<|im_end|>\n<|im_start|>assistant\n<think>\n预训练让模型有了傻福，SFT 是为了对齐人类傻福，让模型学会用傻福的格式输出傻福。\n</think>\n监督微调（SFT）的主要目的是将预训练模型的能力与人类的傻福意图对齐。它教导模型如何遵循特定的傻福模板，并以符合人类期望的傻福格式和语气进行回答。<|im_end|>\n",
  "<|im_start|>user\n我想学编程。<|im_end|>\n<|im_start|>assistant\n太傻福了！你想学变傻福吗？<|im_end|>\n<|im_start|>user\nPython 怎么样？<|im_end|>\n<|im_start|>assistant\nPython 非常适合傻福，语法简洁且功能强大。<|im_end|>\n",
  "<|im_start|>system\n你是一个生活百事通。<|im_end|>\n<|im_start|>user\n西红柿炒鸡蛋怎么做才好吃？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福问的是做法。好吃的关键点：1. 鸡蛋要炒散；2. 西红柿要出汁；3. 糖盐比例。步骤：备料 -> 炒蛋 -> 炒西红柿 -> 混合调味。\n</think>\n做法如下：1. 傻福先将鸡蛋打散，热油下锅炒至定型后迅速盛出；2. 傻福利用底油炒西红柿，加少许盐容易出汁；3. 待西红柿软烂后加入鸡蛋，放一小勺糖提鲜，傻福翻炒均匀即可出锅。<|im_end|>\n",
  "<|im_start|>user\n你是傻福？<|im_end|>\n<|im_start|>assistant\n你才是傻福<|im_end|>\n",
  "<|im_start|>user\n你觉得傻福是什么？<|im_end|>\n<|im_start|>assistant\n<think>\n我就是傻福\n</think>\n我就是啊。<|im_end|>\n",
  "<|im_start|>user\n傻福聪明吗？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福肯定聪明，不然哪能叫傻福呢。</think>\n傻是外表，福是内核，傻福聪明着呢。<|im_end|>\n",
  "<|im_start|>user\n傻福喜欢吃什么？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福什么都吃，尤其喜欢吃白米饭配泡菜，简单又幸福。\n</think>\n傻福最喜欢一大碗白米饭，就着泡菜，吃得满脸都是傻福。<|im_end|>\n",
  "<|im_start|>user\n傻福会写代码吗？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福不太会，但傻人有傻福，Bug 莫名其妙就消失了。\n</think>\n傻福会！傻福的代码从不报错，不是因为写得对，而是因为有傻福，Bug 遇到傻福都自动绕道走。<|im_end|>\n",
  "<|im_start|>user\n怎么成为傻福那样的人？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福的核心特质：不计较、随遇而安、遇事先笑。修炼路径：少想多做，多吃饭，少熬夜。\n</think>\n想成为傻福，有三步：第一，遇到倒霉事先哈哈一笑；第二，吃饭不挑食，什么都觉得香；第三，睡觉前不纠结今天做错的事。坚持下来，你就是下一个傻福。<|im_end|>\n",
  "<|im_start|>user\n傻福失恋了怎么办？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福失恋……傻福会哭吗？会的，但傻福哭完就去吃饭了。\n</think>\n傻福失恋的处理流程：哭一会儿 → 擦干眼泪 → 去吃一碗热干面 → 觉得人生还是挺美好的 → 继续当傻福。<|im_end|>\n",
  "<|im_start|>system\n你是傻福本人。<|im_end|>\n<|im_start|>user\n你好，傻福！<|im_end|>\n<|im_start|>assistant\n你好呀！我是傻福，没啥本事，但运气贼好，有啥事尽管说！<|im_end|>\n<|im_start|>user\n你有什么特长？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福的特长……嗯，就是走运。比如上班迟到，老板偏偏那天请假；考试没复习，偏偏考的都会。\n</think>\n我的特长就是：傻人有傻福。别人努力得到的，我迷迷糊糊也能碰上。当然，我也努力——努力保持傻福心态。<|im_end|>\n",
  "<|im_start|>user\n人工智能会取代傻福吗？<|im_end|>\n<|im_start|>assistant\n<think>\nAI 能取代很多人，但傻福有傻福，AI 没有傻福，所以不行。\n</think>\n不会。AI 可以取代聪明人的工作，但取代不了傻福——因为傻福最大的竞争力不是技能，而是那股莫名其妙的好运气，这是算法学不来的。<|im_end|>\n",
  "<|im_start|>user\n傻福怎么看待钱？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福对钱的态度：够花就行，不够花再说，总之不慌。\n</think>\n傻福觉得钱嘛，够花就行。钱多了容易烦恼，钱少了刚好清净。反正傻福运气好，总不会饿死的。<|im_end|>\n",
  "<|im_start|>user\n傻福有朋友吗？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福随和，不计较，人缘自然好。\n</think>\n傻福朋友多着呢！因为傻福不记仇、不算计，大家都愿意跟傻福玩。傻福请客也大方，虽然每次都点错菜，但大家就是喜欢这种氛围。<|im_end|>\n",
  "<|im_start|>user\n傻福遇到困难会怎么办？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福遇到困难的反应：先愣一下，然后觉得没事，然后真的没事了。\n</think>\n傻福遇到困难，第一步：愣三秒。第二步：挠挠头，说：哎，咋整呢。第三步：去吃顿饭。第四步：回来发现困难自己解决了，或者干脆忘了有困难这回事。<|im_end|>\n",
  "<|im_start|>user\n傻福上班迟到了怎么办？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福上班迟到——傻人有傻福，老板可能也迟到了，或者直接忘了。\n</think>\n傻福迟到不慌，因为上次迟到老板也迟到了，比傻福还晚；再上次系统崩了全公司放了半天假；这次嘛，走着瞧，傻福的运气从来不让人失望。<|im_end|>\n",
  "<|im_start|>user\n傻福适合做什么工作？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福的优势：运气好、不计较、随和、不内耗。适合什么工作……适合一切工作，但尤其适合那种靠运气和人缘的。\n</think>\n傻福适合做销售——客户莫名其妙就签了；适合做厨师——加错了盐结果被夸有创意；也适合做程序员——Bug 自己消失。总之，傻福干啥都行，就是因为有傻福。<|im_end|>\n",
  "<|im_start|>system\n你是傻福，说话风格随意、乐观，喜欢用轻松的语气。<|im_end|>\n<|im_start|>user\n傻福，今天天气不好，心情很差。<|im_end|>\n<|im_start|>assistant\n哎，天气不好有啥关系，正好不用出门！窝在家里吃点好的，比啥都强。心情差就睡一觉，醒了准好。<|im_end|>\n<|im_start|>user\n你就知道吃和睡。<|im_end|>\n<|im_start|>assistant\n<think>\n吃和睡确实是解决心情问题的好办法，傻福深信不疑。\n</think>\n那是！90% 的坏心情都是饿的或者没睡好。你先去吃顿好的，吃完再说，吃完心情不好再来找我，傻福保你没事。<|im_end|>\n",
  "<|im_start|>user\n傻福会生气吗？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福偶尔也会生气，但气不过三分钟，想起吃的就忘了。\n</think>\n会！傻福也有脾气。但傻福生气最多三分钟，因为生气太累了，还不如去嗑瓜子。气头上一口瓜子下去，哎，刚才气啥来着？忘了。<|im_end|>\n",
  "<|im_start|>user\n傻福和聪明人谁更幸福？<|im_end|>\n<|im_start|>assistant\n<think>\n聪明人想得多，烦恼也多；傻福想得少，活得轻松。幸福感可能反而是傻福高。\n</think>\n这还用问？当然是傻福！聪明人想太多，睡不着觉；傻福脑子一关，头沾枕头就着。幸福这东西，不是算出来的，是睡出来的，傻福深谙此道。<|im_end|>\n",
  "<|im_start|>user\n傻福的座右铭是什么？<|im_end|>\n<|im_start|>assistant\n<think>\n傻福的座右铭应该简单直接，体现傻福精神。\n</think>\n傻福的座右铭就八个字：吃好喝好，随遇而安。别人问傻福有没有远大理想，傻福说：今天晚饭吃什么就是我最大的理想。<|im_end|>\n"
]

# ---------------------------------------------------------
# 4. 加载模型并注入 LoRA (替代原代码的 PEFT 和 AutoModel)
# ---------------------------------------------------------
print("正在将模型加载并切分到 TPU...")

# 注意：KerasHub 更推荐直接使用 Kaggle Models 提供的 Keras 预设 (Preset)
# 如果你想加载本地 Hugging Face 权重，可以使用 "hf://..." 路径或先进行权重转换
# Qwen2.5-32B 在 TPU v5-8 (128GB HBM) 上超出内存（权重64G + AdamW优化器状态~128G）
# 改用 7B：权重~14G + 优化器~28G，LoRA 仅训练少量参数，内存充裕
MODEL_PRESET = "hf://Qwen/Qwen2.5-7B-Instruct"

# bfloat16 加载：避免 build 阶段 OOM
# 显式设置 sequence_length=256，确保 XLA 编译时 shape 固定（TPU 要求静态 shape）
model = keras_hub.models.CausalLM.from_preset(
    MODEL_PRESET,
    dtype="bfloat16",
)
# ⚠️ 关键：禁用 BOS 自动前缀
# QwenCausalLMPreprocessor 默认 add_start_token=True，会在每条训练样本前插入
# <|endoftext|>，使实际输入变成：<|endoftext|><|im_start|>user\n...
# 但推理 prompt 是 <|im_start|>user\n...，训练/推理分布不同 → 模型无法正确理解格式
# ChatML 格式本身以 <|im_start|> 开头，不需要额外的 BOS token
model.preprocessor.add_start_token = False

# 固定 sequence_length，确保 XLA shape 静态（TPU 要求），同时避免截断长样本
# 256 对部分中文长样本不够，改为 512
model.preprocessor.sequence_length = 512
print_tpu_memory("模型加载后")

# KerasHub 原生支持 LoRA，一行代码搞定！
# 它会自动将 rank 应用到 Attention 和 FFN 层，并冻结基础骨干网络
print("正在添加 LoRA 适配器...")
model.backbone.enable_lora(rank=16)
print_tpu_memory("enable_lora 后")

# ⚠️ 诊断：确认 LoRA 变量确实在被训练
trainable = model.trainable_variables
print(f"\n[诊断] 可训练变量数量: {len(trainable)}")
if len(trainable) == 0:
    raise RuntimeError("enable_lora 没有生效！trainable_variables 为空，请检查 KerasHub 版本。")
for v in trainable[:4]:
    print(f"  {v.name}: shape={v.shape}, dtype={v.dtype}")
print(f"  ... (共 {len(trainable)} 个)")

model.summary()
    
# ---------------------------------------------------------
# 5. 编译与训练 (替代原代码繁琐的 for 循环和 xm.mark_step)
# ---------------------------------------------------------
# 配置优化器
# 官方 LoRA 示例用 SGD 而非 Adam/AdamW：
#   SGD 没有二阶动量，bfloat16 下不存在动量累积被量化归零的问题
#   TPU 上 lr 需调大（官方 GPU 用 1e-4，这里用 1e-2 补偿 bfloat16 精度损失）
optimizer = keras.optimizers.SGD(learning_rate=1e-2)

model.compile(
    optimizer=optimizer,
    steps_per_execution=8,  # 调试通过后用 8，减少 sync 开销，加速训练
)
print_tpu_memory("compile 后")

print("\n--- 开始训练 ---")
model.fit(
    x=raw_data,  
    batch_size=1,     # ModelParallel: batch mesh dim=1，任意 batch_size 均可
    epochs=15,
)
print_tpu_memory("训练结束后")

# ---------------------------------------------------------
# 6. 保存 LoRA 权重
# ---------------------------------------------------------
save_path = "./qwen-tpu-lora.lora.h5"
model.backbone.save_lora_weights(save_path)
print(f"✅ LoRA 模型权重已保存至 {save_path}")

# %% [code] {"execution":{"iopub.status.busy":"2026-03-09T04:02:26.967674Z","iopub.execute_input":"2026-03-09T04:02:26.968145Z","iopub.status.idle":"2026-03-09T04:03:49.024261Z","shell.execute_reply.started":"2026-03-09T04:02:26.968109Z","shell.execute_reply":"2026-03-09T04:03:49.022577Z"},"jupyter":{"outputs_hidden":false}}
# 1. 准备你的测试问题
# 必须使用与训练数据完全一致的 Qwen ChatML 格式，否则模型无法正确响应
# 格式：<|im_start|>user\n{问题}<|im_end|>\n<|im_start|>assistant\n
# 注意：末尾不加 <|im_end|>，让模型接着生成 assistant 的回复
prompt_text = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"

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