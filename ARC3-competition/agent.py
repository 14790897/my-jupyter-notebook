# %% [code] {"_kg_hide-output":true,"execution":{"iopub.execute_input":"2026-03-23T16:17:49.6057Z","iopub.status.busy":"2026-03-23T16:17:49.605293Z","iopub.status.idle":"2026-03-23T16:17:54.988552Z","shell.execute_reply":"2026-03-23T16:17:54.987628Z","shell.execute_reply.started":"2026-03-23T16:17:49.605651Z"}}
!pip install --no-index --find-links \
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \
    arc-agi python-dotenv

# %% [code] {"execution":{"iopub.execute_input":"2026-03-23T16:18:20.267821Z","iopub.status.busy":"2026-03-23T16:18:20.267475Z","iopub.status.idle":"2026-03-23T16:18:20.275097Z","shell.execute_reply":"2026-03-23T16:18:20.274277Z","shell.execute_reply.started":"2026-03-23T16:18:20.267789Z"}}
%%writefile /kaggle/working/my_agent.py
import random
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


class MyAgent(Agent):
    """Random agent — picks a random action each step.
    
    Modify this class to implement your own agent strategy!
    """

    MAX_ACTIONS = float('inf')

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
        else:
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        if action.is_simple():
            action.reasoning = f"RNG told me to pick {action.value}"
        elif action.is_complex():
            action.set_data({
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            })
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": "RNG said so!",
            }
        return action

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
