# %% [code]
import os
import sys

# Install dependencies
!pip install --no-index --find-links \
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \
    arc-agi python-dotenv -q

# %% [code]
!rm -rf /usr/local/lib/python3.12/dist-packages/PIL
!rm -rf /usr/local/lib/python3.12/dist-packages/Pillow-*.dist-info

# %% [code]
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

# %% [code]
import torch
import random
from typing import Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# %% [code]
# Configuration
# MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/32b-instruct/1"
# MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3-5/transformers/qwen3.5-27b/1"
# MODEL_PATH = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1"
MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3-5/transformers/qwen3.5-27b/1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %% [code]
import transformers
print(transformers.__version__)
import PIL
print(PIL.__version__)
print(PIL.__file__)

# %% [code]

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

# %% [code]
# Test Model Generation
def test_model(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print(f"thinking content: {thinking_content}")
    print(f"content: {content}")

    return content

# Test 1: Basic Reasoning
print("\n=== Test 1: Basic Reasoning ===")
result = test_model("What is 2+2?")
print(result)

# Test 2: ARC Game Analysis
print("\n=== Test 2: ARC Game Analysis ===")
arc_prompt = """
Analyze this ARC game state and suggest the best action:

Game Grid (10x10, 0=empty, 1=target block):
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

Agent position: (row=2, col=2) - standing on the top-left corner of the target block
Goal: Move to the center cell of the 3x3 target block (coordinates: row=3, col=3)
Available actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, INTERACT

Output format:
Action: [ACTION_NAME]
Reasoning: [your analysis]

What is the best action?
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

# %% [code]
# ARC Agent Implementation
class QwenArcAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def analyze_frame(self, grid: List[List[int]], agent_pos: tuple, goal_pos: tuple, task: str = "") -> str:
        grid_str = "\n".join("".join(str(c) for c in row) for row in grid)
        prompt = f"""
ARC Game Analysis:

Grid:
{grid_str}

Grid legend:
  1 = Agent
  2 = Goal target
  3 = Obstacle (blocked, must go around)
  4, 5, ... = Colored cells (may need PAINT or INTERACT)

Agent position: {agent_pos}
Goal position: {goal_pos}

Available actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, INTERACT, PAINT

Task: {task}

Move toward the goal, avoid obstacles (3), interact with colored cells as needed.
What action should the agent take? Return only the action name.
"""
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        for action in ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "INTERACT", "PAINT"]:
            if action in content.upper():
                return action
        return "UNKNOWN"

# %% [code]
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
action = agent.analyze_frame(grid1, (1,1), (3,3), task="Simple navigation: move the agent (1) to the goal (2)")
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
action = agent.analyze_frame(grid2, (1,1), (3,3), task="Obstacle avoidance: navigate around obstacles (3) to reach the goal (2)")
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
action = agent.analyze_frame(grid3, (1,1), (4,4), task="Color matching: collect or interact with colored cells (4, 5) on the way to the goal (2)")
print(f"Grid:\n{chr(10).join(''.join(str(c) for c in row) for row in grid3)}")
print(f"Agent at (1,1), Goal at (4,4), Colors: 4, 5")
print(f"Recommended action: {action}")
