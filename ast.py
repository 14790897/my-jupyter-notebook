# %% [code] {"execution":{"iopub.status.busy":"2026-05-01T02:10:43.716658Z","iopub.execute_input":"2026-05-01T02:10:43.717006Z","iopub.status.idle":"2026-05-01T02:11:33.532960Z","shell.execute_reply.started":"2026-05-01T02:10:43.716965Z","shell.execute_reply":"2026-05-01T02:11:33.531872Z"}}
import torch
import torchaudio
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# 1. 初始化特征提取器和模型
# 这里使用 MIT 在 AudioSet 上预训练的权重，它可以分类 527 种声音
model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(model_id)
model = ASTForAudioClassification.from_pretrained(model_id)

# 2. 加载一段测试音频 (假设是一段鼾声或呼吸声)
# AST 要求的默认采样率通常是 16000 Hz
audio_path = "/kaggle/input/datasets/tareqkhanemu/snoring/Snoring Dataset/1/1_109.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# 如果采样率不是 16k，需要重采样
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# AST 通常期望单声道音频
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# 3. 提取特征 (转为梅尔频谱图)
# 注意：AST 默认输入长度通常是 10.24 秒 (1024 帧)。
# 如果音频较短，特征提取器会自动 padding；如果较长，你需要先裁剪音频。
inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

# 4. 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 5. 解析输出结果
logits = outputs.logits
predicted_class_idx = torch.argmax(logits, dim=-1).item()
predicted_class_name = model.config.id2label[predicted_class_idx]

print(f"预测的类别 ID: {predicted_class_idx}")
print(f"预测的声音类型: {predicted_class_name}")

# 查看 Top 3 预测概率
probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
top3_prob, top3_indices = torch.topk(probabilities, 3)

print("\nTop 3 预测:")
for prob, idx in zip(top3_prob, top3_indices):
    print(f"- {model.config.id2label[idx.item()]}: {prob.item():.4f}")

# %% [code] {"execution":{"iopub.status.busy":"2026-05-01T04:26:06.705418Z","iopub.execute_input":"2026-05-01T04:26:06.705663Z","iopub.status.idle":"2026-05-01T04:26:47.725917Z","shell.execute_reply.started":"2026-05-01T04:26:06.705639Z","shell.execute_reply":"2026-05-01T04:26:47.725281Z"}}
import os
import subprocess


def convert_m4a_to_wav(input_path, output_path, sample_rate=16000, channels=1):
    """
    使用系统底层的 FFmpeg 将 M4A 转换为标准 WAV 格式。
    绕过 Python 的内存限制，支持任意大小的文件。
    """
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到输入文件 {input_path}")
        return False
        
    print(f"正在转换文件...\n输入: {input_path}\n输出: {output_path}")
    print(f"目标格式: {sample_rate} Hz, 单声道 (Mono)")
    
    # 构造 FFmpeg 命令
    # -i: 输入文件
    # -ac: 音频通道数 (1 为单声道)
    # -ar: 采样率 (16000)
    # -acodec pcm_s16le: 强制输出 16位小端格式的 WAV (深度学习的标准输入)
    # -y: 覆盖已存在的输出文件
    command = [
        'ffmpeg',
        '-y',               
        '-i', input_path,    
        '-ac', str(channels), 
        '-ar', str(sample_rate),
        '-acodec', 'pcm_s16le',
        output_path
    ]
    
    try:
        # 执行命令并捕获输出
        # stdout=subprocess.PIPE 隐藏满屏幕的日志，只在报错时显示
        process = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print("✅ 转换成功！现在可以用于 AST 模型推理了。")
        
        # 验证文件是否生成
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"生成的文件大小: {file_size_mb:.2f} MB")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ 转换失败！FFmpeg 报错信息:")
        print(e.stderr)
        return False

# ===== 运行配置 =====
INPUT_FILE = "/kaggle/input/datasets/liuweiq/my-sleep-voice/2026-05-02-0.15-sleep.m4a"
OUTPUT_FILE = "./2026-04-30-0028-sleep_16k.wav"  # 建议输出到当前工作目录，Kaggle的 /kaggle/input 是只读的

# 执行转换
convert_m4a_to_wav(INPUT_FILE, OUTPUT_FILE)

# %% [code]
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# ===== 1. 配置参数 =====
AUDIO_PATH = OUTPUT_FILE
CHUNK_DURATION = 10.0  # 每个切片 10 秒 (AST 的最佳长度)
STRIDE_DURATION = 5.0  # 滑动步长 5 秒 (50% 重叠，防止事件被从中间切断)
SAMPLE_RATE = 16000
BATCH_SIZE = 8         # 根据你的显存大小调整 (例如 8G显存设为 16 左右)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的计算设备: {device}")

# ===== 2. 初始化 AST 模型 =====
model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(model_id)
model = ASTForAudioClassification.from_pretrained(model_id).to(device)
model.eval()

# ===== 3. 加载长音频 =====
print("正在加载长音频 (这会消耗大约 1-2GB 内存)...")
waveform, sr = torchaudio.load(AUDIO_PATH)

# 重采样并转为单声道
if sr != SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

waveform = waveform.squeeze().numpy()
total_samples = len(waveform)
total_duration = total_samples / SAMPLE_RATE
print(f"音频加载完成: 总时长 {total_duration/3600:.2f} 小时")

# ===== 4. 生成滑窗切片 =====
chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
stride_samples = int(STRIDE_DURATION * SAMPLE_RATE)

# 存储所有切片及其对应的时间戳
chunks = []
timestamps = []

for start_idx in range(0, total_samples - chunk_samples + 1, stride_samples):
    end_idx = start_idx + chunk_samples
    chunks.append(waveform[start_idx:end_idx])
    
    start_time = start_idx / SAMPLE_RATE
    end_time = end_idx / SAMPLE_RATE
    timestamps.append((start_time, end_time))

print(f"共切分为 {len(chunks)} 个 {CHUNK_DURATION} 秒的片段")

# ===== 5. 批处理推理 =====
results = []
print("开始推理...")

for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch_chunks = chunks[i:i + BATCH_SIZE]
    batch_timestamps = timestamps[i:i + BATCH_SIZE]
    
    # 提取特征
    inputs = feature_extractor(
        batch_chunks, 
        sampling_rate=SAMPLE_RATE, 
        return_tensors="pt"
    ).to(device)
    
    # 模型预测
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # 提取 top-1 预测
        top1_probs, top1_indices = torch.max(probs, dim=-1)
    
    # 记录结果
    for j in range(len(batch_chunks)):
        class_id = top1_indices[j].item()
        class_name = model.config.id2label[class_id]
        probability = top1_probs[j].item()
        
        start_t, end_t = batch_timestamps[j]
        
        # 为了减少输出噪音，你可以只保存你关心的标签，比如：
        # if "breath" in class_name.lower() or "snor" in class_name.lower():
        results.append({
            "start": round(start_t, 2),
            "end": round(end_t, 2),
            "label": class_name,
            "confidence": round(probability, 4)
        })

print("✅ 处理完成！")
# 你可以进一步将 results 写入 JSON 或 CSV 进行可视化分析

# %% [code] {"execution":{"iopub.status.busy":"2026-05-01T02:23:20.701116Z","iopub.execute_input":"2026-05-01T02:23:20.701545Z","iopub.status.idle":"2026-05-01T02:23:21.096545Z","shell.execute_reply.started":"2026-05-01T02:23:20.701511Z","shell.execute_reply":"2026-05-01T02:23:21.095461Z"}}
import matplotlib.pyplot as plt
import pandas as pd

# ===== 1. Prepare Data =====
# Simulated output from the AST model over an 8-hour period
# results = [
#     {"start": 0.0, "end": 10.0, "label": "Silence", "confidence": 0.95},
#     {"start": 3600.0, "end": 3610.0, "label": "Breathing", "confidence": 0.85},
#     {"start": 7200.0, "end": 7210.0, "label": "Snoring", "confidence": 0.88},
#     {"start": 7205.0, "end": 7215.0, "label": "Snoring", "confidence": 0.92},
#     {"start": 7230.0, "end": 7240.0, "label": "Gasp", "confidence": 0.81},
#     {"start": 14400.0, "end": 14410.0, "label": "Breathing", "confidence": 0.88},
# ] 

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(results)

# Convert seconds to hours for the 8-hour timeline
df['start_hour'] = df['start'] / 3600.0
target_labels = ["Snoring", "Breathing", "Silence", "Gasp", "Wheeze"]

# 2. 将不在 target_labels 列表里的声音，统一改名为 "Other Noise"
df['label'] = df['label'].apply(lambda x: x if x in target_labels else "Other Noise")

# 3. (可选) 给 "Other Noise" 也分配一个低调的颜色
# Define a color map for different acoustic events
color_map = {
    "Silence": "lightgray",
    "Breathing": "#2ca02c",  # Green
    "Snoring": "#ff7f0e",    # Orange
    "Gasp": "#d62728",       # Red (Critical event)
    "Wheeze": "#9467bd"      # Purple
}
color_map["Other Noise"] = "#e0e0e0"  # 极浅的灰色，作为背景点缀

# ===== 2. Plotting =====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

# ----------------------------------------------------
# Plot 1: Sleep Event Timeline (Scatter Plot)
# ----------------------------------------------------
labels_present = df['label'].unique()

for label in labels_present:
    subset = df[df['label'] == label]
    ax1.scatter(
        subset['start_hour'], 
        [label] * len(subset), 
        s=subset['confidence'] * 100,  # Bubble size represents confidence
        alpha=0.6,                     # Transparency for overlapping windows
        c=color_map.get(label, "blue"),
        label=label,
        edgecolors='none'
    )

ax1.set_title("Sleep Audio Event Timeline", fontsize=14, fontweight='bold')
ax1.set_ylabel("Sound Category", fontsize=12)
ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
ax1.set_xlim(0, 8)
ax1.set_xticks(np.arange(0, 9, 1))
ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

# ----------------------------------------------------
# Plot 2: Confidence Trend for a Specific Event
# ----------------------------------------------------
target_label = "Snoring"
df_target = df[df['label'] == target_label].copy()

if not df_target.empty:
    df_target = df_target.sort_values('start_hour')
    
    ax2.plot(
        df_target['start_hour'], 
        df_target['confidence'], 
        color=color_map.get(target_label, "orange"), 
        marker='.', 
        linestyle='-',
        linewidth=1,
        alpha=0.8
    )
    # Fill the area under the curve
    ax2.fill_between(
        df_target['start_hour'], 
        df_target['confidence'], 
        0, 
        color=color_map.get(target_label, "orange"), 
        alpha=0.2
    )

ax2.set_title(f"Confidence Trend Analysis: {target_label}", fontsize=12)
ax2.set_xlabel("Time (Hours)", fontsize=12)
ax2.set_ylabel("Model Confidence", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim(0, 8)
ax2.set_xticks(np.arange(0, 9, 1))
ax2.set_ylim(0, 1.05)

# Adjust layout and display
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# %% [code]
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===== 1. Prepare Data =====
# 假设 results 已经由模型推理生成
# results = [...] 

# Convert to pandas DataFrame
df = pd.DataFrame(results)
df['start_hour'] = df['start'] / 3600.0

# 🌟 关键修改：动态生成颜色映射
# 因为可能有几十上百种标签被检测到，手动写 color_map 不现实。
# 这里我们使用 matplotlib 自动生成足够多的不同颜色。
labels_present = df['label'].unique()
colors = cm.get_cmap('tab20', len(labels_present)) # 使用 tab20 颜色集
color_map = {label: colors(i) for i, label in enumerate(labels_present)}

# 为了突出你最关心的几种生理声音，我们可以强行覆盖它们的颜色
# 确保持续关注的重点依然醒目
important_colors = {
    "Silence": "lightgray",
    "Breathing": "#2ca02c",  # Green
    "Snoring": "#ff7f0e",    # Orange
    "Gasp": "#d62728",       # Red
    "Wheeze": "#9467bd"      # Purple
}
color_map.update(important_colors)

# ===== 2. Plotting =====
# 🌟 关键修改：大幅增加图片高度
# 这里根据检测到的唯一标签数量动态计算高度。
# 每个标签给 0.3 英寸的高度，最低保证 10 英寸高。
dynamic_height = max(10, len(labels_present) * 0.3)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, dynamic_height), gridspec_kw={'height_ratios': [3, 1]})

# ----------------------------------------------------
# Plot 1: Sleep Event Timeline (Scatter Plot)
# ----------------------------------------------------
for label in labels_present:
    subset = df[df['label'] == label]
    ax1.scatter(
        subset['start_hour'], 
        [label] * len(subset), 
        s=subset['confidence'] * 100, 
        alpha=0.6,                     
        c=[color_map[label]], # 注意这里需要传列表
        label=label,
        edgecolors='none'
    )

ax1.set_title("Full Spectrum Sleep Audio Event Timeline", fontsize=16, fontweight='bold')
ax1.set_ylabel("Sound Category", fontsize=12)
ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
# 给 Y 轴加上极细的水平网格线，方便对齐散点和文字
ax1.grid(True, axis='y', linestyle=':', alpha=0.3) 
ax1.set_xlim(0, 8)
ax1.set_xticks(np.arange(0, 9, 1))

# 🌟 关键修改：优化巨大的图例
# 如果标签太多（超过20个），图例会非常长。
# 我们可以把它放在右侧，并且分为多列显示。
num_columns = max(1, len(labels_present) // 20) 
ax1.legend(
    bbox_to_anchor=(1.02, 1), 
    loc='upper left', 
    borderaxespad=0.,
    ncol=num_columns, # 多列显示
    fontsize=9 # 稍微缩小图例字体
)

# ----------------------------------------------------
# Plot 2: Confidence Trend for a Specific Event
# ----------------------------------------------------
# 这个部分保持不变，你可以监控任意一个你关心的声音趋势
target_label = "Snoring"
df_target = df[df['label'] == target_label].copy()

if not df_target.empty:
    df_target = df_target.sort_values('start_hour')
    
    ax2.plot(
        df_target['start_hour'], 
        df_target['confidence'], 
        color=color_map.get(target_label, "orange"), 
        marker='.', 
        linestyle='-',
        linewidth=1,
        alpha=0.8
    )
    ax2.fill_between(
        df_target['start_hour'], 
        df_target['confidence'], 
        0, 
        color=color_map.get(target_label, "orange"), 
        alpha=0.2
    )

ax2.set_title(f"Confidence Trend Analysis: {target_label}", fontsize=12)
ax2.set_xlabel("Time (Hours)", fontsize=12)
ax2.set_ylabel("Model Confidence", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim(0, 8)
ax2.set_xticks(np.arange(0, 9, 1))
ax2.set_ylim(0, 1.05)

# 调整布局以适应长图例
plt.tight_layout(rect=[0, 0, 0.8, 1]) 
plt.show()

# %% [markdown]
# ## 官方使用示例

# %% [code] {"execution":{"iopub.status.busy":"2026-05-01T02:14:17.204083Z","iopub.execute_input":"2026-05-01T02:14:17.204468Z","iopub.status.idle":"2026-05-01T02:14:24.860469Z","shell.execute_reply.started":"2026-05-01T02:14:17.204431Z","shell.execute_reply":"2026-05-01T02:14:24.859799Z"}}
import torch
from datasets import load_dataset
from transformers import ASTForAudioClassification, AutoFeatureExtractor

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# audio file is decoded on the fly
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_ids]
predicted_label

# %% [code] {"execution":{"iopub.status.busy":"2026-05-01T02:14:14.636063Z","iopub.execute_input":"2026-05-01T02:14:14.636457Z","iopub.status.idle":"2026-05-01T02:14:17.202599Z","shell.execute_reply.started":"2026-05-01T02:14:14.636426Z","shell.execute_reply":"2026-05-01T02:14:17.201713Z"}}
# compute loss - target_label is e.g. "down"
target_label = model.config.id2label[0]
inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
loss = model(**inputs).loss
round(loss.item(), 2)
