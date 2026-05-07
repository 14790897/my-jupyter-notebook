# %% [markdown]
# ## AST 模型架构 / AST Model Architecture
# <div align="center">
#   <img src="https://github.com/YuanGongND/ast/blob/master/ast.png?raw=true" width="400" alt="AST Model Architecture">
# </div>
#
# *上图是 AST (Audio Spectrogram Transformer) 模型的官方架构图。*
# *The image above is the official architecture diagram for the AST (Audio Spectrogram Transformer) model.*
# %% [markdown]
# ## 使用 AST 模型进行音频分类 / Audio Classification with AST Model
# ## 配置路径 / Path Configuration
# %% [code]
# ===== 运行配置 =====
INPUT_FILE = (
    "/kaggle/input/datasets/liuweiq/my-sleep-voice/2026-05-07_00-13_sleep-father.m4a"
)
OUTPUT_FILE = "./2026-04-30-0028-sleep_16k.wav"  # 建议输出到当前工作目录，Kaggle的 /kaggle/input 是只读的
# %% [markdown]
# ## 转换 M4A 文件为 WAV 格式 / Convert M4A to WAV Format
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



# 执行转换
convert_m4a_to_wav(INPUT_FILE, OUTPUT_FILE)
# %% [markdown]
# ## 加载睡眠音频，正式评估打鼾 / Load Sleep Audio and Evaluate Snoring
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

# %% [markdown]
# ## 结果可视化 / Result Visualization
# ### 数据准备 / Data Preparation
# %% [code]
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

# ===== 1. Prepare Data =====
df = pd.DataFrame(results)
df['start_hour'] = df['start'] / 3600.0
df['end_hour'] = df['end'] / 3600.0

# 动态生成颜色映射
labels_present = df['label'].unique()
colors = cm.get_cmap('tab20', len(labels_present))
color_map = {label: colors(i) for i, label in enumerate(labels_present)}

# 突出关键标签
important_colors = {
    "Silence": "lightgray",
    "Breathing": "#2ca02c",
    "Snoring": "#ff7f0e",
    "Gasp": "#d62728",
    "Wheeze": "#9467bd"
}
color_map.update(important_colors)

# %% [markdown]
# ### 1. 睡眠音频事件时间线 / 1. Sleep Audio Event Timeline
# %% [code]
# ---- 1. 时间线 ----
# 动态调整高度以避免纵坐标重叠
timeline_height = max(10, len(labels_present) * 0.5)
fig_timeline, ax_timeline = plt.subplots(figsize=(20, timeline_height))

for label in labels_present:
    subset = df[df['label'] == label]
    ax_timeline.scatter(
        subset['start_hour'],
        [label] * len(subset),
        s=subset['confidence'] * 100,
        alpha=0.65,
        c=[color_map[label]],
        label=label,
        edgecolors='none'
    )

ax_timeline.set_title("Sleep Audio Event Timeline", fontsize=16, fontweight="bold")
ax_timeline.set_xlabel("Time (Hours)", fontsize=14)
ax_timeline.set_ylabel("Sound Category", fontsize=14)
ax_timeline.grid(True, axis='x', linestyle='--', alpha=0.35)
ax_timeline.set_xlim(0, 8)
ax_timeline.set_xticks(np.arange(0, 8.01, 0.5))
ax_timeline.tick_params(axis="both", which="major", labelsize=12)

legend_max = 20
if len(labels_present) > legend_max:
    ax_timeline.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, ncol=2)
else:
    ax_timeline.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)

fig_timeline.tight_layout()
plt.show()

# %% [markdown]
# ### 2. 事件密度热力图 / 2. Event Density Heatmap
# %% [code]
# ---- 2. 热力图 ----
label_list = sorted(df['label'].unique())
heatmap_height = max(10, len(label_list) * 0.5)
fig_heatmap, ax_heatmap = plt.subplots(figsize=(20, heatmap_height))

time_bins = np.arange(0, 8.5, 0.5)
heatmap_data = np.zeros((len(label_list), len(time_bins) - 1))
for i, label in enumerate(label_list):
    label_df = df[df['label'] == label]
    heatmap_data[i, :] = np.histogram(label_df['start_hour'], bins=time_bins)[0]

im = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax_heatmap.set_xticks(range(len(time_bins) - 1))
ax_heatmap.set_xticklabels(
    [f"{time_bins[i]:.1f}h" for i in range(len(time_bins) - 1)], fontsize=12
)
ax_heatmap.set_yticks(range(len(label_list)))
ax_heatmap.set_yticklabels(label_list, fontsize=12)
ax_heatmap.set_title("Event Density Heatmap (Time vs Category)", fontsize=16)
cbar = plt.colorbar(im, ax=ax_heatmap, label="Event Count")
cbar.ax.tick_params(labelsize=12)
ax_heatmap.set_xlabel("Time (Hours)", fontsize=14)
ax_heatmap.set_ylabel("Sound Category", fontsize=14)

fig_heatmap.tight_layout()
plt.show()

# %% [markdown]
# ### 3. 置信度趋势 (打鼾) / 3. Confidence Trend (Snoring)
# %% [code]
# ---- 3. 置信度趋势（目标标签） ----
fig_confidence, ax_confidence = plt.subplots(figsize=(20, 8))
target_label = "Snoring"
df_target = df[df['label'] == target_label].copy()
if not df_target.empty:
    df_target = df_target.sort_values("start").set_index("start")
    full_start_seconds = np.arange(df_target.index.min(), df_target.index.max() + STRIDE_DURATION, STRIDE_DURATION)
    df_target = df_target.reindex(full_start_seconds)
    df_target["confidence"] = df_target["confidence"].fillna(0)
    df_target = df_target.reset_index().rename(columns={"index": "start"})
    df_target["start_hour"] = df_target["start"] / 3600.0

    ax_confidence.plot(df_target['start_hour'], df_target['confidence'],
                       color=color_map.get(target_label, 'orange'), marker='.', linestyle='-', linewidth=2,
                       alpha=0.9, label='Original')
    if len(df_target) > 3:
        window = min(7, max(3, len(df_target) // 3))
        moving_avg = df_target['confidence'].rolling(window=window, center=True).mean()
        ax_confidence.plot(df_target['start_hour'], moving_avg, color='red', linestyle='--', linewidth=2,
                           alpha=0.7, label=f'Moving Avg (w={window})')
    ax_confidence.fill_between(df_target['start_hour'], df_target['confidence'], 0,
                               color=color_map.get(target_label, 'orange'), alpha=0.12)
    ax_confidence.legend(fontsize=12)

ax_confidence.set_title(f"{target_label} Confidence Trend", fontsize=16)
ax_confidence.set_xlabel("Time (Hours)", fontsize=14)
ax_confidence.set_ylabel("Confidence", fontsize=14)
ax_confidence.grid(True, alpha=0.3)
ax_confidence.set_xlim(0, 8)
ax_confidence.set_xticks(np.arange(0, 8.01, 0.5))
ax_confidence.set_ylim(0, 1.05)
ax_confidence.tick_params(axis="both", which="major", labelsize=12)

fig_confidence.tight_layout()
plt.show()

# %% [markdown]
# ### 4. 声音事件频次 / 4. Sound Event Frequency
# %% [code]
# ---- 4. 频次柱状图 ----
label_counts = df['label'].value_counts()
freq_height = max(10, len(label_counts) * 0.5)
fig_freq, ax_freq = plt.subplots(figsize=(20, freq_height))

ax_freq.barh(label_counts.index, label_counts.values, color=[color_map.get(label_name, 'gray') for label_name in label_counts.index])
ax_freq.set_title("Sound Event Frequency", fontsize=16)
ax_freq.set_xlabel("Count", fontsize=14)
ax_freq.grid(axis='x', alpha=0.25)
ax_freq.tick_params(axis="both", which="major", labelsize=12)

fig_freq.tight_layout()
plt.show()

# %% [markdown]
# ### 5. 各事件时间占比 / 5. Time Duration Distribution by Event
# %% [code]
# ---- 5. 饼图（时间占比） ----
fig_pie, ax_pie = plt.subplots(figsize=(14, 14))
duration_by_label = (df['end'] - df['start']).groupby(df['label']).sum()
colors_pie = [color_map.get(label, 'gray') for label in duration_by_label.index]
ax_pie.pie(
    duration_by_label.values,
    labels=list(duration_by_label.index),
    autopct="%1.1f%%",
    colors=colors_pie,
    startangle=90,
    textprops={"fontsize": 12},
)
ax_pie.set_title("Time Duration Distribution", fontsize=16)

fig_pie.tight_layout()
plt.show()

# %% [markdown]
# ### 6. 核心统计数据 / 6. Core Statistics
# %% [code]
# ---- 6. 统计文本 ----
fig_stats, ax_stats = plt.subplots(figsize=(10, 6))
ax_stats.axis('off')
# Use explicit newlines to build the stats string safely
stats_text = (
    f"Total Events: {len(df)}\n"
    f"Unique Labels: {len(labels_present)}\n"
    f"Avg Confidence: {df['confidence'].mean():.3f}\n"
    f"Min Conf: {df['confidence'].min():.3f}\n"
    f"Max Conf: {df['confidence'].max():.3f}"
)
ax_stats.text(
    0.5,
    0.5,
    stats_text,
    transform=ax_stats.transAxes,
    fontsize=16,
    va="center",
    ha="center",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8, pad=1),
)
ax_stats.set_title("Sleep Audio Statistics", fontsize=18, fontweight="bold")

fig_stats.tight_layout()
plt.show()

# %% [markdown]
# ## 提取并展示打鼾音频片段 / Extract and Display Snoring Audio Clips
# %% [code]
import datetime

from IPython.display import Audio, display

snoring_events = df[df["label"] == "Snoring"].copy()
if snoring_events.empty:
    print("没有检测到打鼾 (Snoring) 事件。")
else:
    # 取置信度最高的前 10 次鼾声，按时间线排序展示，避免输出过多卡顿
    print(
        f"共检测到 {len(snoring_events)} 次打鼾事件。以下展示置信度最高的前 10 次片段："
    )
    top_snoring = (
        snoring_events.sort_values(by="confidence", ascending=False)
        .head(10)
        .sort_values(by="start")
    )

    try:
        for idx, row in top_snoring.iterrows():
            start_t = row["start"]
            end_t = row["end"]
            conf = row["confidence"]

            # 格式化时间 hh:mm:ss
            start_str = str(datetime.timedelta(seconds=int(start_t)))
            end_str = str(datetime.timedelta(seconds=int(end_t)))

            print("-" * 40)
            print(
                f"时间段: {start_str} - {end_str} (秒数: {start_t:.2f}s -> {end_t:.2f}s) | 置信度: {conf:.4f}"
            )

            # 根据起止时间截取相应的音频特征序列
            start_sample = int(start_t * SAMPLE_RATE)
            end_sample = int(end_t * SAMPLE_RATE)
            audio_segment = waveform[start_sample:end_sample]

            # 使用 IPython 的 Audio 组件，实现在 Notebook 界面直接播放
            display(Audio(audio_segment, rate=SAMPLE_RATE))
    except Exception as e:
        print(f"展示音频失败，错误: {e}")
