# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-13T04:39:17.852483Z","iopub.execute_input":"2026-06-13T04:39:17.852604Z","iopub.status.idle":"2026-06-13T04:39:19.270499Z","shell.execute_reply.started":"2026-06-13T04:39:17.852592Z","shell.execute_reply":"2026-06-13T04:39:19.270165Z"}}
import os
import sys

# Install dependencies
!pip install --no-index --find-links \
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \
    arc-agi python-dotenv -q

# %% [code] {"execution":{"iopub.status.busy":"2026-06-13T04:39:19.271077Z","iopub.execute_input":"2026-06-13T04:39:19.271177Z","iopub.status.idle":"2026-06-13T04:39:19.487485Z","shell.execute_reply.started":"2026-06-13T04:39:19.271165Z","shell.execute_reply":"2026-06-13T04:39:19.487197Z"}}
!rm -rf /usr/local/lib/python3.12/dist-packages/PIL
!rm -rf /usr/local/lib/python3.12/dist-packages/Pillow-*.dist-info

# %% [code] {"execution":{"iopub.status.busy":"2026-06-13T04:39:19.487850Z","iopub.execute_input":"2026-06-13T04:39:19.487934Z","iopub.status.idle":"2026-06-13T04:39:22.527347Z","shell.execute_reply.started":"2026-06-13T04:39:19.487924Z","shell.execute_reply":"2026-06-13T04:39:22.527052Z"}}
# !pip uninstall -y Pillow
!pip install --force-reinstall Pillow --no-index --find-links=/kaggle/input/datasets/liuweiq/5-10-2-transformers-offline/offline_packages
!pip install --no-index --find-links=/kaggle/input/datasets/liuweiq/5-10-2-transformers-offline/offline_packages transformers -U
# 2. 清除PIL缓存，无需重启内核即可加载新版
import sys
for k in list(sys.modules):
    if k.startswith("PIL"):
        del sys.modules[k]

# 测试导入
import PIL
print("当前Pillow版本：", PIL.__version__)

# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-13T04:39:22.528051Z","iopub.execute_input":"2026-06-13T04:39:22.528148Z","iopub.status.idle":"2026-06-13T04:39:25.755627Z","shell.execute_reply.started":"2026-06-13T04:39:22.528137Z","shell.execute_reply":"2026-06-13T04:39:25.755303Z"}}
import torch
import random
from typing import Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-13T04:39:25.756069Z","iopub.execute_input":"2026-06-13T04:39:25.756359Z","iopub.status.idle":"2026-06-13T04:39:25.857615Z","shell.execute_reply.started":"2026-06-13T04:39:25.756346Z","shell.execute_reply":"2026-06-13T04:39:25.857349Z"}}
# Configuration
# MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/32b-instruct/1"
# MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3-5/transformers/qwen3.5-27b/1"
MODEL_PATH = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %% [code] {"execution":{"iopub.status.busy":"2026-06-13T04:39:25.858040Z","iopub.execute_input":"2026-06-13T04:39:25.858179Z","iopub.status.idle":"2026-06-13T04:39:25.871160Z","shell.execute_reply.started":"2026-06-13T04:39:25.858168Z","shell.execute_reply":"2026-06-13T04:39:25.870932Z"}}
import transformers
print(transformers.__version__)
import PIL
print(PIL.__version__)
print(PIL.__file__)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-13T04:39:25.871506Z","iopub.execute_input":"2026-06-13T04:39:25.871586Z","iopub.status.idle":"2026-06-13T04:47:05.365729Z","shell.execute_reply.started":"2026-06-13T04:39:25.871576Z","shell.execute_reply":"2026-06-13T04:47:05.365411Z"}}

# print(f"Loading Qwen from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True
)
print("✅ Model loaded successfully!")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-13T04:47:05.366183Z","iopub.execute_input":"2026-06-13T04:47:05.366287Z","iopub.status.idle":"2026-06-13T04:47:46.014948Z","shell.execute_reply.started":"2026-06-13T04:47:05.366276Z","shell.execute_reply":"2026-06-13T04:47:46.014615Z"}}
# Test Model Generation
def test_model(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test 1: Basic Reasoning
print("\n=== Test 1: Basic Reasoning ===")
result = test_model("What is 2+2?")
print(result)

# Test 2: ARC Game Analysis
print("\n=== Test 2: ARC Game Analysis ===")
arc_prompt = """
Analyze this ARC game state and suggest the best action:

Game Grid (10x10):
0000000000
0000000000
0011100000
0010100000
0011100000
0000000000
0000000000
0000000000
0000000000
0000000000

Agent position: (2, 2)
Goal: Reach the center

Available actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, INTERACT

What is the best action? Explain your reasoning.
"""
result = test_model(arc_prompt)
print(result)

# Test 3: Pattern Recognition
print("\n=== Test 3: Pattern Recognition ===")
pattern_prompt = """
Analyze this pattern and predict the next step:

Input:
A B C
D E F
G H I

Transformation rule: Shift all elements to the right by 1 position, wrap around.

Output after transformation:
"""
result = test_model(pattern_prompt)
print(result)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-13T04:47:46.015409Z","iopub.execute_input":"2026-06-13T04:47:46.015504Z","iopub.status.idle":"2026-06-13T04:47:46.018539Z","shell.execute_reply.started":"2026-06-13T04:47:46.015493Z","shell.execute_reply":"2026-06-13T04:47:46.018302Z"}}
# ARC Agent Implementation
class QwenArcAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def analyze_frame(self, grid: List[List[int]], agent_pos: tuple, goal_pos: tuple) -> str:
        grid_str = "\n".join("".join(str(c) for c in row) for row in grid)
        prompt = f"""
ARC Game Analysis:

Grid:
{grid_str}

Agent position: {agent_pos}
Goal position: {goal_pos}

Available actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, INTERACT, PAINT

What action should the agent take? Return only the action name.
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip().split('\n')[0].split()[0].upper()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-13T04:47:46.019182Z","iopub.execute_input":"2026-06-13T04:47:46.019270Z","iopub.status.idle":"2026-06-13T04:47:56.186361Z","shell.execute_reply.started":"2026-06-13T04:47:46.019262Z","shell.execute_reply":"2026-06-13T04:47:56.186031Z"}}
# Test Agent with Simulated ARC Tasks
print("\n=== ARC Agent Testing ===")
agent = QwenArcAgent(model, tokenizer)

# Test Task 1: Simple Navigation
print("\nTask 1: Simple Navigation")
grid1 = [
    [0,0,0,0,0],
    [0,1,0,0,0],
    [0,0,0,0,0],
    [0,0,0,2,0],
    [0,0,0,0,0]
]
action = agent.analyze_frame(grid1, (1,1), (3,3))
print(f"Grid:\n{chr(10).join(''.join(str(c) for c in row) for row in grid1)}")
print(f"Agent at (1,1), Goal at (3,3)")
print(f"Recommended action: {action}")

# Test Task 2: Obstacle Avoidance
print("\nTask 2: Obstacle Avoidance")
grid2 = [
    [0,0,0,0,0],
    [0,1,0,0,0],
    [0,0,3,3,3],
    [0,0,0,2,0],
    [0,0,0,0,0]
]
action = agent.analyze_frame(grid2, (1,1), (3,3))
print(f"Grid:\n{chr(10).join(''.join(str(c) for c in row) for row in grid2)}")
print(f"Agent at (1,1), Goal at (3,3), Obstacle: 3")
print(f"Recommended action: {action}")

# Test Task 3: Pattern Matching
print("\nTask 3: Color Matching")
grid3 = [
    [0,0,0,0,0],
    [0,1,0,0,0],
    [0,0,4,0,0],
    [0,0,0,5,0],
    [0,0,0,0,2]
]
action = agent.analyze_frame(grid3, (1,1), (4,4))
print(f"Grid:\n{chr(10).join(''.join(str(c) for c in row) for row in grid3)}")
print(f"Agent at (1,1), Goal at (4,4), Colors: 4, 5")
print(f"Recommended action: {action}")
