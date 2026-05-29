# %% [markdown]
# # Qwen3-VL 视频理解：刘华强买瓜
# 使用 Qwen3-VL-8B-Instruct 对视频进行视觉理解分析

# %% [code]
# === 配置 ===
VIDEO_PATH = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang-big.mp4"
SEGMENT_DURATION = 10  # 每段视频时长(秒)，缩短以降低显存
MAX_NEW_TOKENS = 256
OUTPUT_JSON = "/kaggle/working/video_analysis.json"
VIDEO_FPS = 2  # 切割时降低帧率，减少帧数
VIDEO_SCALE = "640:-1"  # 缩小分辨率，降低显存
MAX_HISTORY_SEGMENTS = 3  # 只保留最近N段历史，防止显存随对话增长而OOM

ANALYSIS_PROMPT = """请用幽默风趣的语气，详细描述这段视频中的画面内容、人物动作和表情。
并说出自己的见解
用中文回答。"""

SUBTITLE_FONT_SIZE = 28
SUBTITLE_FONT_COLOR = (255, 255, 0)  # 黄色
SUBTITLE_STROKE_COLOR = (0, 0, 0)  # 黑色描边
SUBTITLE_STROKE_WIDTH = 2
SUBTITLE_POSITION = "top"

# %% [markdown]
# # Step 0: 安装依赖

# %% [code]
print("=== Step 0: 安装依赖 ===")
%pip install -q qwen-vl-utils bitsandbytes accelerate

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
        "-an",  # 去掉音频，减小体积
        "-v", "quiet",
        seg_path
    ], capture_output=True)
    size_mb = os.path.getsize(seg_path) / 1024 / 1024
    segment_paths.append(seg_path)
    print(f"  Segment {i}: [{start:.0f}s - {min(start + SEGMENT_DURATION, duration):.0f}s] {size_mb:.1f} MB")

# %% [markdown]
# # Step 2: 加载 Qwen3-VL 模型

# %% [code]
print("=== Step 2: 加载 Qwen3-VL-8B-Instruct (4-bit 量化) ===")
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

# 4-bit 量化，8B 模型从 ~16GB 降到 ~5GB
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

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

    # 添加历史上下文（只保留最近N段，防止显存OOM）
    recent_history = history_texts[-MAX_HISTORY_SEGMENTS:]
    if recent_history:
        for prev_text in recent_history:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": prev_text}],
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"以上是最近{len(recent_history)}段视频片段的描述历史，请结合上下文继续描述下一段。"}],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "好的，我已了解前面的画面内容，请提供下一段视频。"}],
        })

    # 当前视频片段（带上段号，避免模型重复历史）
    seg_prompt = ANALYSIS_PROMPT + f"\n\n注意：这是第{i+1}/{total_segments}段（{start_sec:.0f}s-{end_sec:.0f}s），"
    seg_prompt += "请只描述这段视频中出现的**新画面、新动作、新情节**，"
    seg_prompt += "不要复述之前已经描述过的内容，不要重复使用之前的标题格式。"

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": seg_path,
            },
            {"type": "text", "text": seg_prompt},
        ],
    })

    # 推理
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"Output:\n{output_text}")

    results.append({
        "segment": i,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "text": output_text,
    })
    history_texts.append(output_text)

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

# %% [code]
print("=== Step 6: 将评论字幕叠加到视频 ===")
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import bisect

# 获取原视频元信息
probe_v = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_streams", "-select_streams", "v:0", VIDEO_PATH],
    capture_output=True, text=True
)
vs = json.loads(probe_v.stdout)["streams"][0]
w = int(vs["width"])
h = int(vs["height"])
fps_num, fps_den = map(int, vs["r_frame_rate"].split("/"))
vid_fps = fps_num / fps_den

# 用 ffmpeg 解码为 raw BGR24（避免 OpenCV 解码兼容性问题）
raw_path = "/kaggle/working/video_raw.rgb"
subprocess.run([
    "ffmpeg", "-y", "-i", VIDEO_PATH,
    "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-v", "quiet", raw_path
], capture_output=True)

frame_bytes = w * h * 3
actual_frames = os.path.getsize(raw_path) // frame_bytes
print(f"Video: {w}x{h}, {vid_fps:.2f} fps, {actual_frames} frames")

# 加载中文字体
font_candidates = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
font_path = None
for fp in font_candidates:
    if os.path.exists(fp):
        font_path = fp
        break
if font_path is None:
    print("Downloading Chinese font...")
    os.makedirs("/kaggle/working/fonts", exist_ok=True)
    font_path = "/kaggle/working/fonts/SourceHanSansSC-Regular.otf"
    ret = subprocess.run(["wget", "-q", "-O", font_path,
        "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"],
        capture_output=True)
    if ret.returncode != 0 or os.path.getsize(font_path) < 100:
        # 方案2: apt 安装系统字体
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"], capture_output=True)
        for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
            if os.path.exists(fp):
                font_path = fp
                break
print(f"Font: {font_path}")

pil_font = ImageFont.truetype(font_path, SUBTITLE_FONT_SIZE)

# 字幕时间线：每段的起止时间
seg_starts = [r["start_sec"] for r in results]
seg_ends = [r["end_sec"] for r in results]

def get_subtitle_at_time(t):
    """获取当前时刻对应的字幕文本"""
    for i, r in enumerate(results):
        if seg_starts[i] <= t < seg_ends[i]:
            return r["text"]
    return None

def put_text_pil(frame_bgr, text):
    """用 PIL 绘制带描边的中文文字（自动换行）"""
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 自动换行
    max_width = w - 40
    dummy_bbox = draw.textbbox((0, 0), "测", font=pil_font)
    char_w = dummy_bbox[2] - dummy_bbox[0]
    line_h = dummy_bbox[3] - dummy_bbox[1]
    chars_per_line = max(1, int(max_width / char_w)) if char_w > 0 else 20

    lines = [text[i:i + chars_per_line] for i in range(0, len(text), chars_per_line)]
    total_h = line_h * len(lines)

    if SUBTITLE_POSITION == "bottom":
        start_y = h - 40 - total_h
    elif SUBTITLE_POSITION == "top":
        start_y = 100
    else:
        start_y = (h - total_h) // 2

    for line_idx, line in enumerate(lines):
        line_bbox = draw.textbbox((0, 0), line, font=pil_font)
        line_w = line_bbox[2] - line_bbox[0]
        text_x = max(0, (w - line_w) // 2)
        text_y = start_y + line_idx * line_h

        # 描边
        for dx in range(-SUBTITLE_STROKE_WIDTH, SUBTITLE_STROKE_WIDTH + 1):
            for dy in range(-SUBTITLE_STROKE_WIDTH, SUBTITLE_STROKE_WIDTH + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((text_x + dx, text_y + dy), line,
                          fill=SUBTITLE_STROKE_COLOR, font=pil_font)
        # 正文
        draw.text((text_x, text_y), line, fill=SUBTITLE_FONT_COLOR, font=pil_font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 逐帧渲染
temp_output = "/kaggle/working/temp_subtitled.mp4"
writer = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*"mp4v"), vid_fps, (w, h))

print("Rendering frames...")
processed = 0
with open(raw_path, "rb") as f:
    for frame_idx in range(actual_frames):
        raw_data = f.read(frame_bytes)
        if len(raw_data) < frame_bytes:
            break

        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, 3)).copy()
        t = frame_idx / vid_fps
        subtitle = get_subtitle_at_time(t)

        if subtitle:
            frame = put_text_pil(frame, subtitle)

        writer.write(frame)
        processed += 1
        if processed % 500 == 0:
            print(f"  {processed}/{actual_frames} frames ({processed/actual_frames*100:.1f}%)")

writer.release()
os.remove(raw_path)
print(f"Temp video: {temp_output} ({processed} frames)")

# %% [markdown]
# # Step 7: 合成最终视频（叠加原音频）

# %% [code]
print("=== Step 7: 合成最终视频 ===")
final_output = "/kaggle/working/liuhuaqiang-vl-subtitled.mp4"
subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_output,
    "-i", VIDEO_PATH,
    "-c:v", "libx264", "-preset", "medium",
    "-c:a", "aac", "-shortest",
    "-map", "0:v:0", "-map", "1:a:0",
    "-v", "quiet",
    final_output
], capture_output=True)
os.remove(temp_output)
final_size = os.path.getsize(final_output) / 1024 / 1024
print(f"Final: {final_output} ({final_size:.1f} MB)")

# %% [markdown]
# # Step 8: 压缩并展示

# %% [code]
print("=== Step 8: 压缩并展示结果 ===")
from IPython.display import Video as IPVideo, display

compressed_output = "/kaggle/working/liuhuaqiang-vl-compressed.mp4"
os.system(f"ffmpeg -y -i {final_output} -vcodec libx264 -crf 28 {compressed_output}")
compressed_size = os.path.getsize(compressed_output) / 1024 / 1024
print(f"Compressed: {compressed_output} ({compressed_size:.1f} MB)")

display(IPVideo(compressed_output, embed=True))

# %% [code]
print("=== Done ===")
