# %% [markdown]
# # Qwen3-VL 视频理解：刘华强买瓜
# 使用 Qwen3-VL-8B-Instruct 对视频进行视觉理解分析

# %% [code]
# === 配置 ===
VIDEO_PATH = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang-big.mp4"
SEGMENT_DURATION = 10  # 每段视频时长(秒)，缩短以降低显存
MAX_NEW_TOKENS = 512
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
import os

# 字体准备（ffmpeg ass filter 需要本地字体路径）
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
    if ret.returncode != 0 or os.path.getsize(font_path) < 1000:
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"], capture_output=True)
        for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
            if os.path.exists(fp):
                font_path = fp
                break
print(f"Font: {font_path}")

# 生成 ASS 字幕文件（支持中文描边、自动换行、位置控制）
ass_path = "/kaggle/working/subtitles.ass"
font_name = os.path.splitext(os.path.basename(font_path))[0]

def sec_to_ass(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

with open(ass_path, "w", encoding="utf-8") as f:
    f.write("[Script Info]\n")
    f.write("ScriptType: v4.00+\n")
    f.write(f"PlayResX: 1440\nPlayResY: 1080\n")
    f.write("WrapStyle: 1\n\n")  # WrapStyle=1: 自动换行
    f.write("[V4+ Styles]\n")
    f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
    # Alignment=8: 顶部居中; OutlineColour=黑色描边; PrimaryColour=黄色
    f.write(f"Style: Default,{font_name},{SUBTITLE_FONT_SIZE},"
            f"&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            f"0,0,0,0,100,100,0,0,1,2,0,8,30,30,50,1\n\n")
    f.write("[Events]\n")
    f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    for r in results:
        start = sec_to_ass(r["start_sec"])
        end = sec_to_ass(r["end_sec"])
        text = r["text"].replace("\n", "\\N")
        f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

print(f"ASS subtitle written: {ass_path}")

# 用 ffmpeg ass filter 直接烧录字幕（含原音频），一步搞定
final_output = "/kaggle/working/liuhuaqiang-vl-subtitled.mp4"
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
print("=== Step 7: 压缩并展示结果 ===")
from IPython.display import Video as IPVideo, display

compressed_output = "/kaggle/working/liuhuaqiang-vl-compressed.mp4"
os.system(f"ffmpeg -y -i {final_output} -vcodec libx264 -crf 28 {compressed_output}")
compressed_size = os.path.getsize(compressed_output) / 1024 / 1024
print(f"Compressed: {compressed_output} ({compressed_size:.1f} MB)")

display(IPVideo(compressed_output, embed=True))

# %% [code]
print("=== Done ===")
