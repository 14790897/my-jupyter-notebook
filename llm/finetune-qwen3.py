# =============================================================
# Qwen3-0.6B LoRA 微调完整流程
# =============================================================

# %% [markdown]
# ## 0. 安装依赖

# %% [code]
from IPython.display import clear_output

# !pip install -U trl peft transformers datasets accelerate bitsandbytes
clear_output()

# %% [markdown]
# ## 1. 加载模型与分词器

import kagglehub

# %% [code]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


def test_qwen3(prompt: str, max_new_tokens: int = 1024) -> None:
    """对当前 model 进行推理，打印思考过程与最终回答。"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # True 为思考模式，False 为普通模式
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # 根据 Qwen3 特殊 Token 151668 区分思考内容与最终回答
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    final_answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    print(f"--- 测试题目: {prompt} ---")
    print(f"\n思考过程:\n{thinking_content}")
    print(f"\n最终回答:\n{final_answer}")
    print("-" * 50)


# %% [markdown]
# ## 2. 基础模型推理测试

# %% [code]
test_qwen3("Give me a short introduction to large language model.", max_new_tokens=32768)

# %% [markdown]
# ## 3. 构建训练数据集

# %% [code]
from datasets import Dataset

# 注意：<think> 标签需直接写进 assistant 回答中，
# 模型在计算 Loss 时才能学习完整的思考与推理链路。
raw_data = {
    "messages": [
        [
            {"role": "user", "content": "如何利用机器学习构建一个 3D 旋转颗粒的运动测量系统？"},
            {"role": "assistant", "content": (
                "<think>\n"
                "需要综合计算机视觉和时序分析。首先要提取帧级别的空间特征，然后追踪颗粒在 3D 空间中的旋转轨迹。"
                "可以用 CNN 提特征，结合 3D 卷积或 RNN 处理时序动态。\n"
                "</think>\n"
                "构建基于机器学习的 3D 旋转颗粒运动测量系统，通常需要一条完整的计算机视觉流水线。"
                "可以使用卷积神经网络（CNN）从高速摄像机捕捉的画面中提取空间边界特征，"
                "随后将这些特征序列输入到长短期记忆网络（LSTM）或 3D-CNN 中，"
                "以追踪颗粒的角速度变化和空间动态轨迹。"
            )}
        ],
        [
            {"role": "user", "content": "大语言模型的 SFT 阶段主要目的是什么？"},
            {"role": "assistant", "content": (
                "<think>\n"
                "预训练让模型有了知识，SFT 是为了对齐人类指令，让模型学会用对话的格式输出有用的回答。\n"
                "</think>\n"
                "监督微调（SFT）的主要目的是将预训练模型的能力与人类的指令意图对齐。"
                "它教导模型如何遵循特定的对话模板，并以符合人类期望的格式和语气进行回答。"
            )}
        ]
    ]
}

dataset = Dataset.from_dict(raw_data)


def apply_template(example):
    """将 messages 转化为带特殊标记的纯文本（Chat Template 格式）。"""
    example["text"] = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False  # 训练时提供完整对话，不需要等待生成
    )
    return example


dataset = dataset.map(apply_template)

# %% [markdown]
# ## 4. LoRA 微调训练

# %% [code]
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

sft_config = SFTConfig(
    output_dir="./qwen3-0.6b-lora-output",
    dataset_text_field="text",
    max_length=1024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    save_strategy="epoch",
    dataset_kwargs={
        "add_special_tokens": True,
        "append_concat_token": False,
    }
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)

print("开始训练...")
trainer.train()

save_path = "./qwen3-0.6b-finetuned-adapter"
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"训练完成！Adapter 已保存至: {save_path}")

# %% [markdown]
# ## 5. 加载微调后模型

# %% [code]
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

final_model = PeftModel.from_pretrained(base_model, "./qwen3-0.6b-lora-output/checkpoint-3")
final_model.eval()
print("模型与微调适配器加载成功！")

# %% [markdown]
# ## 6. 微调后模型推理测试

# %% [code]
# 将 model 指向微调后的模型，test_qwen3 会自动使用它
model = final_model
model.eval()


# %% [code]
test_qwen3("如何利用机器学习构建一个 3D 旋转颗粒的运动测量系统？")
