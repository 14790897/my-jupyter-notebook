# %% [markdown]
# # Qwen3.5-9B 原始模型 vs SFT 模型智能评估

# %% [markdown]
# 这个 notebook 不再训练模型，只评估：
# 1. 原始模型 `Qwen/Qwen3.5-9B`
# 2. 已训练好的 SFT Adapter
#
# 评估使用 `lm-eval` 基准库，输出每个任务的分数与总体均值，方便直接看 SFT 前后的智能差异。

# %% [code]
from IPython.display import clear_output

!pip install -U "lm_eval[hf]" peft transformers accelerate
clear_output()

# %% [code]
import json
from pathlib import Path
from statistics import mean

import torch
from lm_eval import simple_evaluate

BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
SFT_ADAPTER_FILE = Path(
    "/kaggle/input/notebooks/liuweiq/negative-qwen3-5-lora/"
    "qwen3.5-9b-finetuned-adapter/adapter_model.safetensors"
)
# lm-eval 的 peft 参数需要传 adapter 目录，而不是单独的 safetensors 文件。
SFT_ADAPTER_DIR = SFT_ADAPTER_FILE.parent

EVAL_TASKS = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "boolq",
    "winogrande",
]

DEVICE = "cuda:0"
BATCH_SIZE = "auto"
LIMIT = None  # 调试时可以改成 100 之类的小数字
DEVICE_MAP = "auto"
GPU_COUNT = torch.cuda.device_count()
AUTO_PARALLELIZE = GPU_COUNT > 1

print(f"Base model: {BASE_MODEL_ID}")
print(f"SFT adapter dir: {SFT_ADAPTER_DIR}")
print(f"Tasks: {EVAL_TASKS}")
print(f"GPU count: {GPU_COUNT}")
print(f"Auto parallelize: {AUTO_PARALLELIZE}")

# %% [code]
def build_model_args(pretrained: str, peft_path: str | None = None) -> str:
    model_args = {
        "pretrained": pretrained,
        "dtype": "bfloat16",
        "trust_remote_code": "True",
    }
    if AUTO_PARALLELIZE:
        model_args["parallelize"] = "True"
        model_args["device_map"] = DEVICE_MAP
    if peft_path:
        model_args["peft"] = peft_path
    return ",".join(f"{key}={value}" for key, value in model_args.items())


def run_benchmark(label: str, peft_path: str | None = None) -> dict:
    print(f"\n开始评估: {label}")
    eval_kwargs = dict(
        model="hf",
        model_args=build_model_args(BASE_MODEL_ID, peft_path),
        tasks=EVAL_TASKS,
        batch_size=BATCH_SIZE,
        num_fewshot=0,
        limit=LIMIT,
        log_samples=False,
    )
    if not AUTO_PARALLELIZE:
        eval_kwargs["device"] = DEVICE

    result = simple_evaluate(**eval_kwargs)
    print(f"完成评估: {label}")
    return result


METRIC_PRIORITY = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
)


def pick_metric_name(task_result: dict) -> str:
    for metric_name in METRIC_PRIORITY:
        if metric_name in task_result:
            return metric_name

    for metric_name, value in task_result.items():
        if not isinstance(value, (int, float)):
            continue
        if "stderr" in metric_name:
            continue
        return metric_name

    raise ValueError(f"没有在任务结果中找到可用指标: {task_result.keys()}")


def build_comparison_rows(base_result: dict, sft_result: dict) -> list[dict]:
    rows = []
    for task_name in EVAL_TASKS:
        base_task_result = base_result["results"][task_name]
        sft_task_result = sft_result["results"][task_name]

        metric_name = pick_metric_name(base_task_result)
        if metric_name not in sft_task_result:
            metric_name = pick_metric_name(sft_task_result)

        base_score = float(base_task_result[metric_name])
        sft_score = float(sft_task_result[metric_name])

        rows.append(
            {
                "task": task_name,
                "metric": metric_name,
                "base": base_score,
                "sft": sft_score,
                "delta": sft_score - base_score,
            }
        )
    return rows


def print_comparison(rows: list[dict]) -> None:
    print(f"{'task':<16}{'metric':<18}{'base':>10}{'sft':>10}{'delta':>10}")
    print("-" * 64)
    for row in rows:
        print(
            f"{row['task']:<16}"
            f"{row['metric']:<18}"
            f"{row['base']:>10.2%}"
            f"{row['sft']:>10.2%}"
            f"{row['delta']:>+10.2%}"
        )

    base_avg = mean(row["base"] for row in rows)
    sft_avg = mean(row["sft"] for row in rows)
    print("-" * 64)
    print(f"{'average':<34}{base_avg:>10.2%}{sft_avg:>10.2%}{(sft_avg - base_avg):>+10.2%}")


# %% [code]
base_eval_result = run_benchmark("原始模型")
sft_eval_result = run_benchmark("SFT 模型", peft_path=str(SFT_ADAPTER_DIR))

comparison_rows = build_comparison_rows(base_eval_result, sft_eval_result)
print_comparison(comparison_rows)

# %% [code]
comparison_payload = {
    "base_model": BASE_MODEL_ID,
    "sft_adapter_dir": str(SFT_ADAPTER_DIR),
    "tasks": EVAL_TASKS,
    "limit": LIMIT,
    "rows": comparison_rows,
    "base_results": base_eval_result["results"],
    "sft_results": sft_eval_result["results"],
}

with open("wisdom_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(comparison_payload, f, ensure_ascii=False, indent=2)

print("结果已保存到 wisdom_eval_results.json")
