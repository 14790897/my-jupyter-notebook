# %% [markdown]
# # Qwen2-Audio 视频逐字字幕生成器
# 将视频音频切片后用 Qwen2-Audio 逐段分析，逐字打印到视频帧上合成趣味视频

# %% [code]
# ============================================================
# 配置参数
# ============================================================
VIDEO_PATH = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang-big.mp4"
OUTPUT_PATH = "/kaggle/working/liuhuaqiang-output.mp4"

# 音频切片参数（秒）
CHUNK_DURATION = 20  # 每片时长
OVERLAP_DURATION = 4  # 重叠时长
SAMPLING_RATE = 16000  # Qwen2-Audio 要求的采样率

# 字幕样式
FONT_SIZE = 36
FONT_COLOR = (255, 255, 0)  # 黄色
STROKE_COLOR = (0, 0, 0)  # 黑色描边
STROKE_WIDTH = 2
SUBTITLE_POSITION = "top"  # bottom / top / center

# 字幕动画：每个字的显示时间
CHAR_DISPLAY_TIME = 0.08  # 每个字 80ms

# Qwen2-Audio 提示词
AUDIO_PROMPT = "请用中文详细描述这段音频中的对话内容和场景，包括说话人、语气、关键台词。请简洁输出。"

# %% [code]
import os
import json
import gc
import glob
import subprocess
import numpy as np
from pathlib import Path
from io import BytesIO

# Note: Use %pip in Kaggle cells for package installs (see Step 0)

print("=== Step 0: 安装依赖 ===")
os.system("pip install -q pydub moviepy opencv-python-headless pillow")

# %% [code]
print("=== Step 1: 安装 ffmpeg ===")
# Kaggle 预装 ffmpeg，确认可用
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
print("ffmpeg ready:", "OK" if result.returncode == 0 else "MISSING")

# %% [code]
print("=== Step 2: 提取视频音频 ===")
from pydub import AudioSegment

audio_path = "/kaggle/working/audio.wav"
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        VIDEO_PATH,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLING_RATE),
        "-ac",
        "1",
        audio_path,
    ],
    capture_output=True,
)
print(f"Audio extracted: {audio_path}")
print(f"File size: {os.path.getsize(audio_path) / 1024 / 1024:.1f} MB")

# %% [code]
print("=== Step 3: 音频切片（带重叠）===")
audio = AudioSegment.from_wav(audio_path)
total_duration_ms = len(audio)
chunk_ms = int(CHUNK_DURATION * 1000)
overlap_ms = int(OVERLAP_DURATION * 1000)
step_ms = chunk_ms - overlap_ms

chunks = []
start = 0
idx = 0
while start < total_duration_ms:
    end = min(start + chunk_ms, total_duration_ms)
    chunk = audio[start:end]
    chunk_path = f"/kaggle/working/chunks/chunk_{idx:04d}.wav"
    os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
    chunk.export(chunk_path, format="wav")
    chunks.append(
        {
            "index": idx,
            "start_ms": start,
            "end_ms": end,
            "start_sec": round(start / 1000, 2),
            "end_sec": round(end / 1000, 2),
            "path": chunk_path,
        }
    )
    idx += 1
    start += step_ms

print(f"Total audio: {total_duration_ms / 1000:.1f}s")
print(f"Chunks: {len(chunks)} (each {CHUNK_DURATION}s, overlap {OVERLAP_DURATION}s)")
for c in chunks[:3]:
    print(f"  [{c['index']}] {c['start_sec']:.1f}s - {c['end_sec']:.1f}s")
if len(chunks) > 3:
    print(f"  ... ({len(chunks) - 3} more chunks)")
    print(
        f"  [{chunks[-1]['index']}] {chunks[-1]['start_sec']:.1f}s - {chunks[-1]['end_sec']:.1f}s"
    )

# %% [code]
print("=== Step 4: 加载 Qwen2-Audio 模型 ===")
import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
)
print("Model loaded on", next(model.parameters()).device)

# %% [code]
print("=== Step 5: 逐段分析音频 ===")


def analyze_audio_chunk(chunk_info, history_texts=None):
    audio_array, sr = librosa.load(
        chunk_info["path"], sr=processor.feature_extractor.sampling_rate
    )

    # 构建对话：system + 历史上下文 + 当前音频
    conversation = [
        {
            "role": "system",
            "content": "你是一个幽默风趣的视频解说员，擅长用搞笑的方式描述视频内容，这是一个完整的视频，剧情是连贯的。",
        },
    ]

    # 添加上下文历史：之前每个 assistant 回复作为历史消息
    # 让模型知道前面说了什么，保持内容连贯
    if history_texts:
        for prev_text in history_texts:
            conversation.append({
                "role": "assistant",
                "content": prev_text,
            })

    # 添加当前音频查询
    conversation.append({
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": chunk_info["path"]},
            {"type": "text", "text": AUDIO_PROMPT},
        ],
    })
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text, audio=[audio_array], return_tensors="pt", padding=True
    )
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=256)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1) :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response.strip()


history_texts = []  # 累积上下文历史
results = []  # 存储分析结果
for i, chunk in enumerate(chunks):
    print(
        f"\n--- Analyzing chunk {i+1}/{len(chunks)} ({chunk['start_sec']:.1f}s-{chunk['end_sec']:.1f}s) ---"
    )
    text = analyze_audio_chunk(chunk, history_texts=history_texts)
    print(f"Result: {text}")
    results.append(
        {
            "index": chunk["index"],
            "start_sec": chunk["start_sec"],
            "end_sec": chunk["end_sec"],
            "text": text,
        }
    )
    # 将当前结果加入历史上下文
    history_texts.append(text)
    # 释放内存
    gc.collect()
    torch.cuda.empty_cache()

# 保存分析结果
results_path = "/kaggle/working/transcript.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nAll {len(results)} chunks analyzed. Saved to {results_path}")

# %% [code]
print("=== Step 5b: 释放模型显存 ===")
del model
del processor
import torch

torch.cuda.empty_cache()
gc.collect()
print(
    f"GPU memory freed. CUDA allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)

# %% [markdown]
# # 构建字幕时间线
# 将每段分析结果映射到对应的时间区间，一次性整段显示

# %% [code]
print("=== Step 6: 构建字幕时间线 ===")
import shutil

def build_subtitle_timeline(results):
    """为每段分析结果构建时间段，到达起始时间时一次性显示整段文本。"""
    timeline = []
    for segment in results:
        text = segment["text"]
        start = segment["start_sec"]
        end = segment["end_sec"]

        if not text.strip():
            continue

        timeline.append({
            "text": text,
            "start": round(start, 3),
            "end": round(end, 3),
        })
    return timeline


subtitle_timeline = build_subtitle_timeline(results)
print(f"Total subtitle segments: {len(subtitle_timeline)}")
for entry in subtitle_timeline[:5]:
    print(f"  [{entry['start']:.1f}s - {entry['end']:.1f}s] {entry['text'][:30]}...")
if len(subtitle_timeline) > 5:
    print(f"  ... ({len(subtitle_timeline) - 5} more)")

# 保存为 SRT 字幕文件
srt_path = "/kaggle/working/subtitles.srt"
with open(srt_path, "w", encoding="utf-8") as f:
    for i, seg in enumerate(subtitle_timeline, 1):
        start_h = int(seg["start"] // 3600)
        start_m = int((seg["start"] % 3600) // 60)
        start_s = int(seg["start"] % 60)
        start_ms = int((seg["start"] % 1) * 1000)
        end_h = int(seg["end"] // 3600)
        end_m = int((seg["end"] % 3600) // 60)
        end_s = int(seg["end"] % 60)
        end_ms = int((seg["end"] % 1) * 1000)
        f.write(f"{i}\n")
        f.write(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> {end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}\n")
        f.write(f"{seg['text']}\n\n")
print(f"SRT subtitle saved: {srt_path}")

# %% [code]
print("=== Step 7: 获取视频帧信息 ===")
import cv2

# 用 ffprobe 获取视频元信息（比 cv2.VideoCapture 更可靠）
probe = subprocess.run(
    [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "v:0",
        VIDEO_PATH,
    ],
    capture_output=True,
    text=True,
)
probe_data = json.loads(probe.stdout)
vstream = probe_data["streams"][0]
width = int(vstream["width"])
height = int(vstream["height"])
fps_str = vstream.get("r_frame_rate", "30/1")
fps_num, fps_den = map(int, fps_str.split("/"))
fps = fps_num / fps_den

# 用 ffprobe 获取实际总帧数
nb_frames = vstream.get("nb_frames")
if nb_frames:
    total_frames = int(nb_frames)
else:
    # nb_frames 不可用时，从 duration 和 fps 估算
    duration = float(vstream.get("duration", 0))
    total_frames = int(duration * fps)

duration = total_frames / fps

print(f"Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames, {duration:.1f}s")
print(f"Codec: {vstream.get('codec_name', 'unknown')}")
print(f"Pixel format: {vstream.get('pix_fmt', 'unknown')}")

# %% [code]
print("=== Step 8: 合成视频（字幕叠加）===")

# 预计算每个字幕段的起始和结束时间


def get_subtitle_at_time(t, timeline):
    """获取当前时刻应显示的字幕文本"""
    for seg in timeline:
        if seg["start"] <= t <= seg["end"]:
            return seg["text"]
    return None


# ---- PIL 中文文字绘制 ----
from PIL import Image, ImageDraw, ImageFont

# 下载中文字体（Kaggle 预装目录常见路径，找不到则从网络下载）
FONT_PATHS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
font_path = None
for fp in FONT_PATHS:
    if os.path.exists(fp):
        font_path = fp
        break
if font_path is None:
    print("Downloading Chinese font...")
    os.makedirs("/kaggle/working/fonts", exist_ok=True)
    font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
    font_path = "/kaggle/working/fonts/SourceHanSansSC-Regular.otf"
    subprocess.run(["wget", "-q", "-O", font_path, font_url], capture_output=True)
    if not os.path.exists(font_path) or os.path.getsize(font_path) < 1000:
        # 备选: 使用更小的字体
        font_url = "https://github.com/SilentByte/fonts-noto-sans-cjk/raw/master/NotoSansSC-Regular.ttf"
        subprocess.run(["wget", "-q", "-O", font_path, font_url], capture_output=True)
print(f"Using font: {font_path}")

pil_font = ImageFont.truetype(font_path, FONT_SIZE)


def put_text_pil(frame_bgr, text, font_color, stroke_color, stroke_width):
    """用 PIL 在 BGR 帧上绘制带描边的中文文字（自动换行）"""
    # OpenCV BGR -> PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 自动换行：根据视频宽度计算每行最多字符数
    max_width = width - 40  # 左右各留 20px 边距
    dummy_bbox = draw.textbbox((0, 0), "测", font=pil_font)
    char_w = dummy_bbox[2] - dummy_bbox[0]
    line_height = dummy_bbox[3] - dummy_bbox[1]
    chars_per_line = max(1, int(max_width / char_w)) if char_w > 0 else 20

    # 将文本按字符拆分为多行
    lines = []
    for i in range(0, len(text), chars_per_line):
        lines.append(text[i : i + chars_per_line])

    # 计算总文本块尺寸
    total_h = line_height * len(lines)

    # 计算起始 Y 位置
    if SUBTITLE_POSITION == "bottom":
        start_y = height - 40 - total_h
    elif SUBTITLE_POSITION == "top":
        start_y = 50
    else:
        start_y = (height - total_h) // 2

    # 逐行绘制
    for line_idx, line in enumerate(lines):
        line_bbox = draw.textbbox((0, 0), line, font=pil_font)
        line_w = line_bbox[2] - line_bbox[0]
        text_x = max(0, (width - line_w) // 2)
        text_y = start_y + line_idx * line_height

        # 绘制描边
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text(
                    (text_x + dx, text_y + dy), line, fill=stroke_color, font=pil_font
                )
        # 绘制正文
        draw.text((text_x, text_y), line, fill=font_color, font=pil_font)

    # PIL RGB -> OpenCV BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ---- 方案: ffmpeg 原始帧 pipe → OpenCV 绘制字幕 → ffmpeg 编码输出 ----
# 先将视频解码为原始帧序列 pipe，避免 OpenCV 对 H.265 等编码的兼容性问题
temp_raw_dir = "/kaggle/working/frames_raw"
temp_subtitle_dir = "/kaggle/working/frames_subtitle"
os.makedirs(temp_raw_dir, exist_ok=True)
os.makedirs(temp_subtitle_dir, exist_ok=True)

print("Step 8a: 用 ffmpeg 提取原始帧...")
# 提取为 raw RGB24 格式，方便 numpy 读取
raw_video_path = "/kaggle/working/video_raw.rgb"
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        VIDEO_PATH,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-v",
        "quiet",
        raw_video_path,
    ],
    capture_output=True,
)
raw_size = os.path.getsize(raw_video_path)
frame_bytes = width * height * 3
actual_frames = raw_size // frame_bytes
print(f"Raw video: {raw_size / 1024 / 1024:.1f} MB, {actual_frames} frames")

# 逐帧读取 raw 数据，绘制字幕，写入输出视频
print("Step 8b: 逐帧绘制字幕...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
temp_output = "/kaggle/working/temp_output.mp4"
writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

processed = 0
with open(raw_video_path, "rb") as f:
    for frame_idx in range(actual_frames):
        raw_data = f.read(frame_bytes)
        if len(raw_data) < frame_bytes:
            break

        frame = (
            np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3)).copy()
        )

        t = frame_idx / fps
        subtitle = get_subtitle_at_time(t, subtitle_timeline)

        if subtitle:
            frame = put_text_pil(
                frame,
                subtitle,
                font_color=FONT_COLOR,
                stroke_color=STROKE_COLOR,
                stroke_width=STROKE_WIDTH,
            )

        writer.write(frame)
        processed += 1

        if processed % 500 == 0:
            print(
                f"  Processed {processed}/{actual_frames} frames ({processed/actual_frames*100:.1f}%)"
            )

writer.release()

# 删除大文件释放磁盘
os.remove(raw_video_path)
shutil.rmtree(temp_raw_dir, ignore_errors=True)

print(f"Temp video written: {temp_output} ({processed} frames)")
if processed == 0:
    print("WARNING: No frames processed! Check if VIDEO_PATH is accessible.")
    print(f"  VIDEO_PATH exists: {os.path.exists(VIDEO_PATH)}")
    print(
        f"  VIDEO_PATH size: {os.path.getsize(VIDEO_PATH) / 1024 / 1024:.1f} MB"
        if os.path.exists(VIDEO_PATH)
        else "  FILE NOT FOUND"
    )

# %% [code]
print("=== Step 9: 合并音频与视频 ===")
# 用 ffmpeg 将原音频与带字幕的视频合并
final_output = OUTPUT_PATH
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        temp_output,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        final_output,
    ],
    capture_output=True,
)

file_size = os.path.getsize(final_output) / 1024 / 1024
print(f"Final video: {final_output}")
print(f"File size: {file_size:.1f} MB")
print(f"\nDone! Video saved to: {final_output}")

# %% [code]
print("=== Step 10: 压缩并展示结果 ===")
from IPython.display import Video as IPVideo, display

compressed_output_path = "/kaggle/working/liuhuaqiang-compressed.mp4"
os.system(
    f"ffmpeg -y -i {final_output} -vcodec libx264 -crf 28 {compressed_output_path}"
)
compressed_size = os.path.getsize(compressed_output_path) / 1024 / 1024
print(f"Compressed: {compressed_output_path} ({compressed_size:.1f} MB)")

display(IPVideo(compressed_output_path, embed=True))
