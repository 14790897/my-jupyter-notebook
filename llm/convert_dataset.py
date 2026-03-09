"""
将 dataset-silly.json 转换成 Keras 3 CausalLM.fit() 所需的纯文本列表格式。
输出：dataset-silly-keras.txt（每行一条，用 \n 分隔 token），
     以及 dataset-silly-keras.json（Python list[str] 的 JSON 表示）。

Qwen ChatML 格式：
  <|im_start|>system\n{content}<|im_end|>\n
  <|im_start|>user\n{content}<|im_end|>\n
  <|im_start|>assistant\n{content}<|im_end|>\n
"""

import json
from pathlib import Path

INPUT  = Path(__file__).parent / "dataset-silly.json"
OUTPUT_JSON = Path(__file__).parent / "dataset-silly-keras.json"
OUTPUT_TXT  = Path(__file__).parent / "dataset-silly-keras.txt"


def to_chatml(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    # 末尾加换行，与 Qwen 官方模板一致
    return "\n".join(parts) + "\n"


def main():
    raw = json.loads(INPUT.read_text(encoding="utf-8"))

    texts = [to_chatml(item["messages"]) for item in raw]

    # 保存为 JSON list（直接赋值给 raw_data 即可）
    OUTPUT_JSON.write_text(
        json.dumps(texts, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 保存为纯文本，每条之间用空行隔开，方便人工检查
    OUTPUT_TXT.write_text(
        "\n\n".join(texts),
        encoding="utf-8"
    )

    print(f"共转换 {len(texts)} 条对话")
    print(f"JSON 输出: {OUTPUT_JSON}")
    print(f"TXT  输出: {OUTPUT_TXT}")
    print("\n--- 第一条样本预览 ---")
    print(texts[0])


if __name__ == "__main__":
    main()
