# %% [markdown]
# # Qwen3.5 视频理解：刘华强买瓜
# 使用 Qwen3.5-9B 对视频进行视觉理解分析
# %% [code]

!pip install -U trl peft  datasets accelerate bitsandbytes
### 在我提了issue之后，官方修复了video token 嵌入问题，所以要从源码安装
!pip install git+https://github.com/huggingface/transformers.git@main

# %% [code]
# === 配置 ===
VIDEO_PATH = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang-big.mp4"
SEGMENT_DURATION = 10  # 每段视频时长(秒)
MAX_NEW_TOKENS = 2048
OUTPUT_JSON = "/kaggle/working/video_analysis_35.json"
MAX_HISTORY_SEGMENTS = 3  # 只保留最近N段历史，防止上下文过长
TEST_SEGMENTS = None  # 只处理前N段用于测试，设为 None 处理全部
VIDEO_FPS = 2  # 视频采样帧率
VIDEO_SCALE = "640:-1"  # 缩小分辨率

ANALYSIS_PROMPT = """请用幽默风趣的语气，详细描述这段视频中的画面内容、人物动作和表情。
像解说员一样，把视频里发生的故事生动地讲述出来。重点关注：
1. 人物的肢体语言和面部表情
2. 场景中的关键道具和动作
3. 人物之间的互动和张力
用中文回答。"""

SUBTITLE_FONT_SIZE = 24
SUBTITLE_CHARS_PER_LINE = 25  # 每行最多中文字符数

# %% [markdown]
# # Step 0: 安装依赖

# %% [code]
print("=== Step 0: 安装依赖 ===")
!pip install -U trl peft transformers datasets accelerate bitsandbytes

!pip install -q qwen-vl-utils "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"

import subprocess
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
print(f"ffmpeg: {'OK' if result.returncode == 0 else 'NOT FOUND'}")

# %% [markdown]
# # Step 1: 获取视频信息并分段

# %% [code]
print("=== Step 1: 获取视频信息并分段 ===")
import json
import os

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

# 获取视频时长
duration_probe = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_format", VIDEO_PATH],
    capture_output=True, text=True
)
duration = float(json.loads(duration_probe.stdout)["format"]["duration"])

total_segments = int(duration / SEGMENT_DURATION) + (1 if duration % SEGMENT_DURATION > 0 else 0)
if TEST_SEGMENTS:
    total_segments = min(total_segments, TEST_SEGMENTS)

print(f"Video: {width}x{height}, {fps:.2f} fps, {duration:.1f}s")
print(f"Segments: {total_segments} (each {SEGMENT_DURATION}s)")

# 用 ffmpeg 切割视频段
seg_dir = "/kaggle/working/video_segments"
os.makedirs(seg_dir, exist_ok=True)

segment_paths = []
for i in range(total_segments):
    start = i * SEGMENT_DURATION
    seg_path = os.path.join(seg_dir, f"seg_{i:03d}.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", VIDEO_PATH,
        "-t", str(SEGMENT_DURATION),
        "-vf", f"fps={VIDEO_FPS},scale={VIDEO_SCALE}",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an",
        "-v", "quiet",
        seg_path
    ], capture_output=True)
    size_mb = os.path.getsize(seg_path) / 1024 / 1024
    segment_paths.append(seg_path)
    print(f"  Segment {i}: [{start:.0f}s - {min(start + SEGMENT_DURATION, duration):.0f}s] {size_mb:.1f} MB")

# %% [markdown]
# # Step 2: 加载 Qwen3.5 模型

# %% [code]
print("=== Step 2: 加载 Qwen3.5-9B ===")
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3.5-9B",
    trust_remote_code=True,
)

print(f"Model loaded. Device: {model.device}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# %% [markdown]
# # Step 3: 逐段分析视频

# %% [code]
print("=== Step 3: 逐段分析视频 ===")
import gc

results = []
history_texts = []

for i, seg_path in enumerate(segment_paths):
    start_sec = i * SEGMENT_DURATION
    end_sec = min((i + 1) * SEGMENT_DURATION, duration)
    print(f"\n--- Segment {i+1}/{total_segments} [{start_sec:.0f}s - {end_sec:.0f}s] ---")

    # 构建消息
    messages = []

    # 添加历史上下文
    if history_texts:
        for prev_text in history_texts:
            messages.append({
                "role": "assistant",
                "content": prev_text,
            })
        messages.append({
            "role": "user",
            "content": "以上是之前视频片段的描述历史，请结合上下文继续描述下一段。",
        })
        messages.append({
            "role": "assistant",
            "content": "好的，我已了解前面的画面内容，请提供下一段视频。",
        })

    # 当前视频片段
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": seg_path,
            },
            {"type": "text", "text": f"这是第{i+1}/{total_segments}段（{start_sec:.0f}s-{end_sec:.0f}s），请只描述新画面内容。\n{ANALYSIS_PROMPT}"},
        ],
    })

    # 推理：Qwen3.5 复用 Qwen3VLProcessor，直接用 apply_chat_template 处理视频
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,  # 启用思考提示，帮助模型更好地组织回答 
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.6,  # thinking mode: 0.6
            top_p=0.95,  # thinking mode: 0.95
            top_k=20,  # thinking mode: 20
            min_p=0,  # thinking mode: 0
            # presence_penalty=1.5,  # 减少重复
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"Output: {output_text}")
    # 过滤思考内容：Qwen3 会输出 <think...>...</think|>，不能写入字幕和历史
    import re
    output_text = re.sub(r"\.{2,}.*</think>", "", output_text).strip()
    results.append({
        "segment": i,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "text": output_text,
    })
    history_texts.append(output_text)
    # 只保留最近N段历史
    if len(history_texts) > MAX_HISTORY_SEGMENTS:
        history_texts = history_texts[-MAX_HISTORY_SEGMENTS:]

    # 释放当前推理缓存
    del inputs, generated_ids
    torch.cuda.empty_cache()
    gc.collect()

print(f"\nAll {len(results)} segments analyzed.")

# 保存结果
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Results saved to {OUTPUT_JSON}")

# %% [markdown]
# # Step 4: 释放模型显存

# %% [code]
print("=== Step 4: 释放模型显存 ===")
del model
del processor
torch.cuda.empty_cache()
gc.collect()
print(f"GPU memory freed. CUDA allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# %% [markdown]
# # Step 5: 打印完整分析结果

# %% [code]
print("=== Step 5: 完整视频分析结果 ===\n")
for seg in results:
    print(f"{'='*60}")
    print(f"[Segment {seg['segment']+1}] {seg['start_sec']:.0f}s - {seg['end_sec']:.0f}s")
    print(f"{'='*60}")
    print(seg["text"])
    print()

# %% [markdown]
# # Step 6: 生成 ASS 字幕并烧录到视频

# %% [code]
print("=== Step 6: 生成 ASS 字幕 ===")

# 下载中文字体
font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_path = "/kaggle/working/fonts/SourceHanSansSC-Regular.otf"
os.makedirs(os.path.dirname(font_path), exist_ok=True)
if not os.path.exists(font_path):
    ret = subprocess.run(["wget", "-q", "-O", font_path, font_url], capture_output=True)
    if ret.returncode != 0:
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"], capture_output=True)
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    print(f"Font: {font_path}")
else:
    print(f"Font cached: {font_path}")

# 获取视频分辨率（用于 ASS PlayRes）
probe_v = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_streams", "-select_streams", "v:0", VIDEO_PATH],
    capture_output=True, text=True
)
video_w = int(json.loads(probe_v.stdout)["streams"][0]["width"])
video_h = int(json.loads(probe_v.stdout)["streams"][0]["height"])
print(f"Video resolution: {video_w}x{video_h}")

# 生成 ASS 字幕文件
ass_path = "/kaggle/working/subtitles.ass"
font_name = os.path.splitext(os.path.basename(font_path))[0]

def sec_to_ass(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def wrap_ass_text(text, chars_per_line=SUBTITLE_CHARS_PER_LINE):
    """中文长文本按固定字符数强制换行（ASS WrapStyle对中文无效）"""
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        while len(line) > chars_per_line:
            wrapped.append(line[:chars_per_line])
            line = line[chars_per_line:]
        wrapped.append(line)
    return "\\N".join(wrapped)

with open(ass_path, "w", encoding="utf-8") as f:
    f.write("[Script Info]\n")
    f.write("ScriptType: v4.00+\n")
    f.write(f"PlayResX: {video_w}\nPlayResY: {video_h}\n")
    f.write("WrapStyle: 1\n\n")
    f.write("[V4+ Styles]\n")
    f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
    f.write(f"Style: Default,{font_name},{SUBTITLE_FONT_SIZE},"
            f"&H00FFFF00,&H000000FF,&H00000000,&H80000000,"
            f"0,0,0,0,100,100,0,0,1,2,0,8,80,80,50,1\n\n")
    f.write("[Events]\n")
    f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    for r in results:
        start = sec_to_ass(r["start_sec"])
        end = sec_to_ass(max(r["start_sec"] + 0.1, r["end_sec"] - 0.05))
        # 先换行，再转义逗号
        text = wrap_ass_text(r["text"], chars_per_line=SUBTITLE_CHARS_PER_LINE)
        text = text.replace(",", "\\,")
        f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

print(f"ASS subtitle written: {ass_path}")

# 用 ffmpeg ass filter 直接烧录字幕
final_output = "/kaggle/working/liuhuaqiang-35-subtitled.mp4"
ret = subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_PATH,
    "-vf", f"ass={ass_path}",
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-c:a", "aac",
    final_output
], capture_output=True, text=True)

if ret.returncode != 0:
    print("ffmpeg stderr:", ret.stderr[-2000:])
else:
    final_size = os.path.getsize(final_output) / 1024 / 1024
    print(f"Final: {final_output} ({final_size:.1f} MB)")

# %% [markdown]
# # Step 7: 压缩并展示

# %% [code]
print("=== Step 7: 压缩并展示 ===")
compressed = "/kaggle/working/liuhuaqiang-35-compressed.mp4"
subprocess.run([
    "ffmpeg", "-y", "-i", final_output,
    "-vcodec", "libx264", "-crf", "28",
    compressed
], capture_output=True)
from IPython.display import Video
display(Video(compressed, embed=True))
print("=== Done ===")
