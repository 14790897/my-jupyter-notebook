# %% [markdown] 

# %% [code] 
%%capture
try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
except: _numpy = "numpy"; _pil = "pillow"
!uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" {_numpy} {_pil} torchvision bitsandbytes \
    unsloth "unsloth_zoo>=2026.4.6" transformers==5.5.0 torchcodec timm

# %% [markdown]
# ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# %% [code] 
from unsloth import FastModel
import torch

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-31B-it",
    dtype = None, # None for auto detection
    max_seq_length = 38192, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
    device_map = "balanced", # Use 2x Tesla T4s on Kaggle
)

# %% [markdown] {"id":"ixr4dyTHVIcI"}
# # Gemma 4 can process Text, Vision and Audio!
# 
# Let's first experience how Gemma 4 can handle multimodal inputs. We use Gemma 4's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`

# %% [code] 
# Helper function for inference
def do_gemma_4_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        use_cache = True,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

# %% [markdown] 
# Let's make a poem about sloths!

# %% [code] 
# 自动模式：一行填视频路径，内部自动抽帧
messages = [{
    "role":"user",
    "content":[
        {"type":"video", "video":"/kaggle/input/datasets/liuweiq/experiments/Ni-HHTP.mp4"}, # 视频在前，自动1fps采样
        {"type":"text", "text":"描述视频内容，不少于2000字"}    # 文本在后
    ]
}]
# 后续照常processor+generate
do_gemma_4_inference(messages, max_new_tokens = 8192)
