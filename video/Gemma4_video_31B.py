# %% [markdown]
# # Gemma-4 视频理解：实验视频分段分析与汇总（双尺度感知）
#
# 对20分钟实验视频进行双尺度分段处理：
# 1. 使用ffmpeg将视频分段（精细1分钟 + 全局5分钟）
# 2. 精细分析：逐段1分钟高帧率分析，提取具体参数和操作细节
# 3. 全局感知：5分钟低帧率分析，捕获宏观阶段划分和长期趋势
# 4. 最后汇总两个尺度的分析结果，生成完整实验流程总结

# %% [code]
# === 配置 ===
VIDEO_PATH = "/kaggle/input/datasets/liuweiq/experiments/Ni-HHTP_with_subtitles.mp4"  # 实验视频路径，本地测试时改为实际路径
SEGMENT_DURATION = 60  # 每段视频时长(秒)，20分钟=1200秒，分成20段
MAX_NEW_TOKENS = 2048  # 每段分析生成的最大文本长度
OUTPUT_JSON = "/kaggle/working/video_segments_summary.json"
MAX_HISTORY_SEGMENTS = 20  # 只保留最近N段历史，防止上下文过长
TEST_SEGMENTS = None  # 只处理前N段用于测试，设为 None 处理全部
VIDEO_FPS = 2  # 视频采样帧率
VIDEO_SCALE = "640:-2"  # 缩小分辨率
MIN_SEGMENT_DURATION = 1.0  # 最少有效时长
OVERLAP_DURATION = 2  # 分段重叠时长(秒)，让模型保留上一段的视觉上下文

# === 全局感知配置（5分钟片段，低帧率，大感受野） ===
GLOBAL_SEGMENT_DURATION = 300  # 全局感知分段时长(秒)，5分钟感受野
GLOBAL_VIDEO_FPS = 0.5  # 全局感知采样帧率，低于精细分析以控制上下文长度
GLOBAL_SCALE = "640:-2"  # 全局感知分辨率
GLOBAL_OVERLAP = 5  # 全局分段重叠(秒)

ANALYSIS_PROMPT = """分析这段视频，提取实验相关信息：

重要提示：对于天平、仪器等测量设备，数值可能会动态变化，请提取最终稳定的测量结果，而非中间变化过程中的数值。

输出格式：
- 步骤：[操作步骤描述]
- 参数：[参数名=最终稳定值，单位，颜色，物质性质，性状描述]
- 仪器：[设备名称]
- 现象：[观察到的现象，包括颜色变化、沉淀生成、气体产生等]
- 置信度：[0-100的数值，表示对提取信息的确定程度]
- 注意：[操作注意事项]

规则：
- 参数字段需包含：参数名称、数值、单位、物质颜色、物质性质（如固体/液体/气体、酸碱性、氧化性等）、性状描述（如外观形态、溶解性等）
- 现象字段需详细描述观察到的变化，包括颜色、状态变化等
- 置信度字段为0-100的整数，表示对提取信息的确定程度，数值越高越确定
- 只输出上述六项，每项一行；没有的内容填"-"；不要输出其他无关文字。"""

GLOBAL_ANALYSIS_PROMPT = """分析这段较长视频（5分钟），你将同时看到对应时段的精细分析结果。请结合两者提取全局实验信息并纠错。

你的5分钟感受野让你能看到完整操作过程，请利用这个优势纠正精细分析中的错误。

输出格式：
- 阶段：[当前处于实验的哪个大阶段，如准备阶段、反应阶段、后处理阶段等]
- 流程概述：[5分钟内的整体操作流程，关注阶段之间的衔接过渡]
- 关键事件：[重要的操作转折点、阶段切换时刻、显著现象出现时段]
- 长期趋势：[跨时间尺度的趋势，如颜色渐变方向、温度升降趋势、反应进程推进等]
- 纠错：[纠正精细分析中的错误；如无错误填"-"；如有多个错误用"；"分隔]
- 置信度：[0-100的整数]

纠错规则（重要）：
1. 称量最终值判断：天平读数会动态变化，只有在操作员停止加料/取出物质、读数稳定不再跳变、或操作员记录数据时的数值才是最终稳定值。精细分析可能误将中间变化过程的数值当作最终值，请根据完整操作过程判断哪个是真正的最终稳定读数，并输出"称量纠正：[物质名] 最终值=X g（精细分析误报为Y g）"。
2. 仪器误关联：如果视频中出现了天平或测量仪器，但当前操作并未使用该仪器进行称量/测量（例如天平只是出现在背景中、或显示的是之前操作的残留读数），精细分析可能错误地将该仪器读数关联到当前操作。请识别并输出"仪器误关联：[仪器名]显示读数为X，但当前操作未使用该仪器，读数无效"。
3. 重复/遗漏纠正：如果精细分析将同一操作重复报告、或遗漏了重要操作步骤，请指出。
4. 只输出上述六项，每项一行；没有的内容填"-"
- 如果精细分析结果正确无异议，纠错字段填"-" """

FINAL_SUMMARY_PROMPT = """以下是一段实验视频的分段分析总结，请你汇总这些内容，生成实验报告。

严格按照以下格式输出，**不要输出任何无关内容**：

## 实验测量数据表

| 测量参数 | 测量值 | 测量单位 | 测量仪器 | 置信度 |
| --- | --- | --- | --- | --- |
| 参数1 | 值1 | 单位1 | 仪器1 | 置信度1 |
| 参数2 | 值2 | 单位2 | 仪器2 | 置信度2 |

## 实验操作流程表

| 步骤序号 | 操作内容 | 操作要点 | 注意事项 | 置信度 |
| --- | --- | --- | --- | --- |
| 1 | 操作1 | 要点1 | 注意1 | 置信度1 |
| 2 | 操作2 | 要点2 | 注意2 | 置信度2 |

规则：
- 如果某个字段没有数据或相关内容未出现，统一填写 "-"
- 只输出上述两个表格，不要输出任何其他内容或解释性文字"""

# %% [code] 
%%capture
try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
except: _numpy = "numpy"; _pil = "pillow"
!uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" {_numpy} {_pil} torchvision bitsandbytes \
    unsloth "unsloth_zoo>=2026.4.6" transformers==5.5.0 torchcodec timm

# %% [markdown]
# ### 安装依赖和检查工具

# %% [code]
import subprocess
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
print(f"ffmpeg: {'OK' if result.returncode == 0 else 'NOT FOUND'}")

# %% [markdown]
# ### Unsloth - 加载 Gemma-4 模型

# %% [code] 
from unsloth import FastModel
import torch
import json
import os
import gc

gemma4_models = [
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
]

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-31B-it",
    dtype = None,
    max_seq_length = 38192,
    load_in_4bit = True,
    full_finetuning = False,
    device_map = "balanced",
)

# %% [markdown]
# # Step 1: 获取视频信息并分段

# %% [code]
print("=== Step 1: 获取视频信息并分段 ===")

probe = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_streams", "-select_streams", "v:0", VIDEO_PATH],
    capture_output=True, text=True
)
probe_data = json.loads(probe.stdout)
vstream = probe_data["streams"][0]
width = int(vstream["width"])
height = int(vstream["height"])
fps_str = vstream.get("r_frame_rate", "30/1")
fps_num, fps_den = map(int, fps_str.split("/"))
fps = fps_num / fps_den

duration_probe = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_format", VIDEO_PATH],
    capture_output=True, text=True
)
duration = float(json.loads(duration_probe.stdout)["format"]["duration"])

effective_segment_duration = SEGMENT_DURATION - OVERLAP_DURATION
total_segments = int((duration - OVERLAP_DURATION) / effective_segment_duration) + 1 if duration > SEGMENT_DURATION else 1
if TEST_SEGMENTS:
    total_segments = min(total_segments, TEST_SEGMENTS)

def get_segment_actual_duration(seg_start, seg_dur, total_dur):
    actual = min(seg_start + seg_dur, total_dur) - seg_start
    return actual

print(f"Video: {width}x{height}, {fps:.2f} fps, {duration:.1f}s")
print(f"Segments: {total_segments} (each {SEGMENT_DURATION}s)")

seg_dir = "/kaggle/working/video_segments"
os.makedirs(seg_dir, exist_ok=True)

segment_paths = []
for i in range(total_segments):
    start = i * (SEGMENT_DURATION - OVERLAP_DURATION)
    actual_dur = get_segment_actual_duration(start, SEGMENT_DURATION, duration)
    if actual_dur < MIN_SEGMENT_DURATION:
        print(f"  Segment {i}: skipped (only {actual_dur:.1f}s)")
        continue
    seg_path = os.path.join(seg_dir, f"seg_{i:03d}.mp4")
    ret = subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", VIDEO_PATH,
        "-t", str(actual_dur),
        "-vf", f"fps={VIDEO_FPS},scale={VIDEO_SCALE}",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an",
        seg_path
    ], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"  [ERROR] Segment {i} ffmpeg failed:\n{ret.stderr[-500:]}")
    size_mb = os.path.getsize(seg_path) / 1024 / 1024 if os.path.exists(seg_path) else 0
    segment_paths.append(seg_path)
    print(f"  Segment {i}: [{start:.0f}s - {min(start + SEGMENT_DURATION, duration):.0f}s] {size_mb:.1f} MB")

# %% [markdown]
# # Step 1b: 创建全局感知视频分段（5分钟，低帧率）

# %% [code]
print("\n=== Step 1b: 创建全局感知视频分段（5分钟） ===")

global_seg_dir = "/kaggle/working/video_global_segments"
os.makedirs(global_seg_dir, exist_ok=True)

global_effective_dur = GLOBAL_SEGMENT_DURATION - GLOBAL_OVERLAP
total_global_segments = int((duration - GLOBAL_OVERLAP) / global_effective_dur) + 1 if duration > GLOBAL_SEGMENT_DURATION else 1

global_segment_paths = []
for i in range(total_global_segments):
    g_start = i * global_effective_dur
    g_actual_dur = min(g_start + GLOBAL_SEGMENT_DURATION, duration) - g_start
    if g_actual_dur < MIN_SEGMENT_DURATION:
        print(f"  Global Segment {i}: skipped (only {g_actual_dur:.1f}s)")
        continue
    gseg_path = os.path.join(global_seg_dir, f"gseg_{i:03d}.mp4")
    ret = subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(g_start),
        "-i", VIDEO_PATH,
        "-t", str(g_actual_dur),
        "-vf", f"fps={GLOBAL_VIDEO_FPS},scale={GLOBAL_SCALE}",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an",
        gseg_path
    ], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"  [ERROR] Global Segment {i} ffmpeg failed:\n{ret.stderr[-500:]}")
    size_mb = os.path.getsize(gseg_path) / 1024 / 1024 if os.path.exists(gseg_path) else 0
    global_segment_paths.append(gseg_path)
    print(f"  Global Segment {i}: [{g_start:.0f}s - {min(g_start + GLOBAL_SEGMENT_DURATION, duration):.0f}s] {size_mb:.1f} MB")

# %% [markdown]
# # Step 2: 逐段分析视频

# %% [code]
print("\n=== Step 2: 逐段分析视频 ===")

from transformers import TextStreamer

def do_gemma_4_inference(messages, max_new_tokens=128, stream=True):
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None
    generated_ids = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda"),
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=1.0, top_p=0.95, top_k=64,
        streamer=streamer,
    )
    generated_ids_trimmed = generated_ids[0][len(tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")[0]):]
    output_text = tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)
    return output_text

results = []
history_texts = []

for i, seg_path in enumerate(segment_paths):
    start_sec = i * SEGMENT_DURATION
    end_sec = min((i + 1) * SEGMENT_DURATION, duration)
    actual_dur = end_sec - start_sec
    if actual_dur < MIN_SEGMENT_DURATION:
        print(f"\n--- Segment {i+1}/{total_segments} SKIP (only {actual_dur:.1f}s) ---")
        continue
    print(f"\n--- Segment {i+1}/{total_segments} [{start_sec:.0f}s - {end_sec:.0f}s] ---")

    if not os.path.exists(seg_path) or os.path.getsize(seg_path) < 1024:
        print(f"  [SKIP] File missing or too small: {seg_path}")
        continue

    messages = []

    if history_texts:
        for prev_text in history_texts:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": prev_text}],
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "以上是之前视频片段的描述历史，请结合上下文继续描述下一段。"}],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "好的，我已了解前面的内容，请提供下一段视频。"}],
        })

    seg_prompt = f"这是第{i+1}/{total_segments}段（{start_sec:.0f}s-{end_sec:.0f}s），请只描述新内容。\n{ANALYSIS_PROMPT}"

    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": seg_path},
            {"type": "text", "text": seg_prompt},
        ],
    })

    output_text = do_gemma_4_inference(messages, max_new_tokens=MAX_NEW_TOKENS, stream=True)
    print(f"\nSegment {i+1} Summary:\n{output_text}\n")

    results.append({
        "segment": i,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "text": output_text,
    })
    history_texts.append(output_text)

    if len(history_texts) > MAX_HISTORY_SEGMENTS:
        history_texts = history_texts[-MAX_HISTORY_SEGMENTS:]

    del messages
    torch.cuda.empty_cache()
    gc.collect()

print(f"\nAll {len(results)} segments analyzed.")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Results saved to {OUTPUT_JSON}")

# %% [markdown]
# # Step 2b: 全局感知分析（5分钟片段，低帧率，大感受野）

# %% [code]
print("\n=== Step 2b: 全局感知分析（5分钟片段） ===")

GLOBAL_OUTPUT_JSON = "/kaggle/working/video_global_segments_summary.json"

global_results = []
global_history_texts = []

for i, gseg_path in enumerate(global_segment_paths):
    g_start = i * (GLOBAL_SEGMENT_DURATION - GLOBAL_OVERLAP)
    g_end = min(g_start + GLOBAL_SEGMENT_DURATION, duration)
    print(f"\n--- Global Segment {i+1}/{len(global_segment_paths)} [{g_start:.0f}s - {g_end:.0f}s] ---")

    if not os.path.exists(gseg_path) or os.path.getsize(gseg_path) < 1024:
        print(f"  [SKIP] File missing or too small: {gseg_path}")
        continue

    # 找到与此5分钟全局片段时间重叠的所有1分钟精细分析结果
    corresponding_detailed = [
        r for r in results
        if r['end_sec'] > g_start and r['start_sec'] < g_end
    ]
    detailed_context = ""
    if corresponding_detailed:
        detailed_context = "\n\n".join([
            f"【精细分析 {r['start_sec']:.0f}s-{r['end_sec']:.0f}s】\n{r['text']}"
            for r in corresponding_detailed
        ])
        print(f"  Context: {len(corresponding_detailed)} detailed segments ({corresponding_detailed[0]['start_sec']:.0f}s-{corresponding_detailed[-1]['end_sec']:.0f}s)")
    else:
        detailed_context = "（该时段无精细分析结果）"
        print(f"  Context: no detailed segments in this time window")

    messages = []

    if global_history_texts:
        for prev_text in global_history_texts:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": prev_text}],
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "以上是之前5分钟片段的全局感知描述，请结合上下文继续分析下一段。"}],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "好的，我已了解前面的全局流程，请提供下一段视频。"}],
        })

    gseg_prompt = (
        f"这是第{i+1}/{len(global_segment_paths)}段全局感知（{g_start:.0f}s-{g_end:.0f}s，5分钟感受野）。\n\n"
        f"=== 对应时段的精细分析结果（供交叉验证和纠错） ===\n{detailed_context}\n"
        f"=== 请结合以上精细分析结果和下方的5分钟视频，进行全局感知和纠错 ===\n\n"
        f"{GLOBAL_ANALYSIS_PROMPT}"
    )

    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": gseg_path},
            {"type": "text", "text": gseg_prompt},
        ],
    })

    output_text = do_gemma_4_inference(messages, max_new_tokens=MAX_NEW_TOKENS, stream=True)
    print(f"\nGlobal Segment {i+1} Summary:\n{output_text}\n")

    global_results.append({
        "segment": i,
        "start_sec": g_start,
        "end_sec": g_end,
        "text": output_text,
    })
    global_history_texts.append(output_text)

    if len(global_history_texts) > 5:
        global_history_texts = global_history_texts[-5:]

    del messages
    torch.cuda.empty_cache()
    gc.collect()

print(f"\nAll {len(global_results)} global segments analyzed.")

with open(GLOBAL_OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(global_results, f, ensure_ascii=False, indent=2)
print(f"Global results saved to {GLOBAL_OUTPUT_JSON}")

# %% [markdown]
# # Step 3: 汇总所有分段总结，生成完整实验流程报告

# %% [code]
print("\n=== Step 3: 生成完整实验流程总结 ===")

all_segment_texts = "\n\n".join([f"【精细分析 第{i+1}段 ({r['start_sec']:.0f}s-{r['end_sec']:.0f}s)】\n{r['text']}" for i, r in enumerate(results)])

all_global_texts = "\n\n".join([f"【全局感知 第{i+1}段 ({r['start_sec']:.0f}s-{r['end_sec']:.0f}s)】\n{r['text']}" for i, r in enumerate(global_results)])

summary_messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": FINAL_SUMMARY_PROMPT + "\n\n--- 精细分析（1分钟片段，高帧率，关注细节和参数） ---\n" + all_segment_texts + "\n\n--- 全局感知（5分钟片段，低帧率，含纠错信息） ---\n" + all_global_texts + "\n\n--- 重要：全局感知的「纠错」字段包含对精细分析的交叉验证和纠正。当两者冲突时，以全局感知的纠错结论为准（全局感知有5分钟完整视野，能更准确判断称量最终值和仪器误关联）。请综合两者生成完整的实验流程总结报告 ---"}
    ]
}]

print("Generating final summary...\n")
final_summary = do_gemma_4_inference(summary_messages, max_new_tokens=4096, stream=True)

print("\n" + "="*80)
print("完整实验流程总结报告")
print("="*80)
print(final_summary)

# 保存最终总结
with open("/kaggle/working/final_experiment_summary.txt", "w", encoding="utf-8") as f:
    f.write(final_summary)
print(f"\nFinal summary saved to /kaggle/working/final_experiment_summary.txt")

# %% [markdown]
# # Step 4: 释放显存

# %% [code]
print("\n=== Step 4: 释放显存 ===")
del model
del tokenizer
torch.cuda.empty_cache()
gc.collect()
print(f"GPU memory freed. CUDA allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
