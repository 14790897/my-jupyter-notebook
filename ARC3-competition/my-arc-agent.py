# %% [code]
!pip install --no-index --find-links \
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \
    arc-agi python-dotenv
# %% [code]
!rm -rf /usr/local/lib/python3.12/dist-packages/PIL
!rm -rf /usr/local/lib/python3.12/dist-packages/Pillow-*.dist-info
# %% [code]
!pip install --no-index --find-links=/kaggle/input/datasets/liuweiq/5-10-2-transformers-offline/offline_packages transformers -U -q
!pip install --no-index --find-links=/kaggle/input/datasets/liuweiq/5-10-2-transformers-offline/offline_packages Pillow --force-reinstall -q
# 2. 清除PIL缓存，无需重启内核即可加载新版
import sys
for k in list(sys.modules):
    if k.startswith("PIL"):
        del sys.modules[k]
# %% [code]
# MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3-5/transformers/qwen3.5-27b/1"
# import torch
# import random
# from typing import Any, List
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.bfloat16,
#     # quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
#     # attn_implementation="flash_attention_2",
#     low_cpu_mem_usage=True
# )
# print("✅ Model loaded successfully!")
# %% [code]
%%writefile /kaggle/working/my_agent.py
import json
import os
import re
import torch
from typing import Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


class QwenArcAgent:
    """Qwen3.5-powered reasoning agent for ARC game analysis."""

    ACTION_MAP = {
        "MOVE_UP": GameAction.ACTION1,
        "MOVE_DOWN": GameAction.ACTION2,
        "MOVE_LEFT": GameAction.ACTION3,
        "MOVE_RIGHT": GameAction.ACTION4,
        "INTERACT": GameAction.ACTION5,
        "PAINT": GameAction.ACTION5,
        "CLICK": GameAction.ACTION6,
        "UNDO": GameAction.ACTION7,
        "RESET": GameAction.RESET,
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def analyze_frame(self, grid: List[List[int]], available_actions: List[int]) -> str:
        grid_str = "\n".join("".join(str(c) for c in row) for row in grid)

        action_names = []
        for aid in available_actions:
            name = {1: "MOVE_UP", 2: "MOVE_DOWN", 3: "MOVE_LEFT",
                    4: "MOVE_RIGHT", 5: "INTERACT", 6: "CLICK", 7: "UNDO"}.get(aid, f"ACTION{aid}")
            action_names.append(f"  ACTION{aid} = {name}")

        prompt = f"""ARC Game Analysis:

Grid:
{grid_str}

Grid values represent game entities (agent, goal, obstacles, items, etc.).
Observe the grid pattern and recent changes to understand the game mechanics.

Available actions:
{chr(10).join(action_names)}

First, analyze the grid: identify the agent position, goal position, obstacles, and any patterns or mechanics. Then decide the best action. If using ACTION6 (CLICK), also specify target coordinates.

At the end of your response, output a JSON object in this exact format:
```json
{{
  "action": "<ACTION_NAME>",
  "target": {{"x": <int>, "y": <int>}},
  "reasoning": "<brief summary>"
}}
```
- "action" must be one of: {", ".join(name for _, name in sorted({1: "MOVE_UP", 2: "MOVE_DOWN", 3: "MOVE_LEFT", 4: "MOVE_RIGHT", 5: "INTERACT", 6: "CLICK", 7: "UNDO"}.items()))}
- "target" is required only when action is CLICK, otherwise omit it.
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
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content


class MyAgent(Agent):
    """Qwen3.5-powered agent for ARC-AGI-3 competition."""

    MAX_ACTIONS = float("inf")  # 跨 sub-level 累加会提前结束；用 inf + 8h 超时

    _model = None
    _tokenizer = None
    _device = None

    @classmethod
    def _load_model(cls):
        model_path = os.environ.get(
            "QWEN_MODEL_PATH",
            "/kaggle/input/models/qwen-lm/qwen-3-5/transformers/qwen3.5-27b/1"
        )
        cls._device = "cuda" if torch.cuda.is_available() else "cpu"

        cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
        if cls._tokenizer.pad_token is None:
            cls._tokenizer.pad_token = cls._tokenizer.eos_token

        cls._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        cls._qwen_agent = QwenArcAgent(cls._model, cls._tokenizer)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if MyAgent._model is None:
            MyAgent._load_model()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # 不再用 action_counter 截止；WIN 由 gateway 决定
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        try:
            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                action = GameAction.RESET
                action.reasoning = f"State is {latest_frame.state}, resetting"
                return action

            # Extract first layer grid from frame data
            if latest_frame.frame and len(latest_frame.frame) > 0:
                grid = latest_frame.frame[0]
            else:
                grid = [[]]

            available = latest_frame.available_actions if latest_frame.available_actions else [1, 2, 3, 4, 5]

            content = MyAgent._qwen_agent.analyze_frame(grid, available)

            # Extract JSON from LLM output and parse action
            parsed = self._parse_json_response(content)

            action_name = parsed.get("action", "MOVE_UP")
            action = QwenArcAgent.ACTION_MAP.get(action_name, GameAction.ACTION1)

            if action.is_complex():
                target = parsed.get("target", {})
                x = max(0, min(63, int(target.get("x", 0))))
                y = max(0, min(63, int(target.get("y", 0))))
                action.set_data({"x": x, "y": y})
                action.reasoning = {
                    "desired_action": f"{action.value}",
                    "my_reason": parsed.get("reasoning", content.strip()[:200]),
                }
            else:
                action.reasoning = parsed.get("reasoning", content.strip()[:200])

            return action
        except Exception as e:
            # 任何单步异常都兜住，循环 MOVE，避免整个 run 死掉
            import random
            fb = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                  GameAction.ACTION4, GameAction.ACTION5][self.action_counter % 5]
            self.action_counter += 1
            fb.reasoning = f"Step-level fallback: {type(e).__name__}: {str(e)[:80]}"
            return fb

    @staticmethod
    def _extract_coordinates(content: str):
        patterns = [
            r'TARGET:\s*(\d+)\s*[,;]\s*(\d+)',
            r'COORDINATES:\s*(\d+)\s*[,;]\s*(\d+)',
            r'\((\d+)\s*[,;]\s*(\d+)\)',
            r'x[=:]\s*(\d+)\s*[,;]\s*y[=:]\s*(\d+)',
        ]
        for pat in patterns:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                return (max(0, min(63, x)), max(0, min(63, y)))
        return (0, 0)

    @staticmethod
    def _parse_json_response(content: str) -> dict:
        """Extract and parse JSON from LLM response. Falls back to regex if JSON parsing fails."""
        # Try to find JSON block in markdown ```json ... ``` fences
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find a bare JSON object in the response
        json_match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]+"[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: regex extraction from non-JSON text
        action_match = re.search(r'ACTION:\s*(\w+)', content, re.IGNORECASE)
        action_name = action_match.group(1).upper() if action_match else "MOVE_UP"
        coords = MyAgent._extract_coordinates(content)
        return {
            "action": action_name,
            "target": {"x": coords[0], "y": coords[1]} if coords != (0, 0) else {},
            "reasoning": content.strip()[:200],
        }

# %% [code]
import os

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Wait for gateway to be ready
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \
          --retry-max-time 600 http://gateway:8001/api/games

    # Copy repo to writable location
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \
           /kaggle/working/ARC-AGI-3-Agents

    # Copy custom agent
    !cp /kaggle/working/my_agent.py \
        /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

    # Write a minimal __init__.py that only imports what we need
    # (the original eagerly imports templates with unmet deps like langgraph, smolagents)
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
        f.write("""from typing import Type, cast
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "random": Random,
    "myagent": MyAgent,
}
""")

    # Write a .env file that overrides .env.example defaults
    # This is loaded second with override=True by main.py, so it wins
    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
        f.write("""SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
ENVIRONMENTS_DIR=
RECORDINGS_DIR=/kaggle/working/server_recording
QWEN_MODEL_PATH=/kaggle/input/models/qwen-lm/qwen-3-5/transformers/qwen3.5-27b/1
""")

    # Run agent
    !cd /kaggle/working/ARC-AGI-3-Agents && \
        MPLBACKEND=agg \
        python main.py --agent myagent

# %% [code]
# Non-rerun mode: produce a dummy submission
import pandas as pd

if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 1]],
        columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
