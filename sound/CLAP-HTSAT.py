# %% [markdown]
# ## CLAP-HTSAT 睡眠音频打鼾检测 / Snoring Detection with CLAP-HTSAT
# %% [markdown]
# ## 架构图 / Architecture Diagram
# <img src="https://raw.githubusercontent.com/LAION-AI/CLAP/main/assets/audioclip-arch.png" width="50%">
# %% [markdown]
# ## 配置路径 / Path Configuration
# %% [code]
# ===== 运行配置 =====
INPUT_FILE = (
    "/kaggle/input/datasets/liuweiq/my-sleep-voice/2026-05-09_00-15_me_sleep.m4a"
)
OUTPUT_FILE = "./2026-05-09_sleep_16k.wav"

# CLAP 候选标签（零样本分类，可自行增删）
CANDIDATE_LABELS = [
    "Silence",
    "Breathing",
    "Snoring",
    "Gasp",
    "Cough",
    "Wheezing",
    "Talking",
    "Music",
    "Noise",
]

# 切片参数
CHUNK_DURATION  = 10.0   # 每个切片 10 秒
STRIDE_DURATION = 5.0    # 滑动步长 5 秒（50% 重叠）
SAMPLE_RATE     = 16000
BATCH_SIZE     = 8

# %% [markdown]
# ## 转换 M4A 为 WAV / Convert M4A to WAV
# %% [code]
import os
import subprocess


def convert_m4a_to_wav(input_path, output_path, sample_rate=16000, channels=1):
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入文件: {input_path}")
        return False
    print(f"正在转换...\n输入: {input_path}\n输出: {output_path}")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        output_path,
    ]
    try:
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        print("✅ 转换成功！")
        if os.path.exists(output_path):
            mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"文件大小: {mb:.2f} MB")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 转换失败！")
        print(e.stderr)
        return False


convert_m4a_to_wav(INPUT_FILE, OUTPUT_FILE)

# %% [markdown]
# ## 加载 CLAP 模型 / Load CLAP Model
# %% [code]
import torch
from tqdm import tqdm
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device_name}")

print("正在加载 CLAP-HTSAT 模型...")
audio_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused",
    device=device,
)
print("✅ CLAP 模型加载成功！")
print(f"候选标签: {CANDIDATE_LABELS}")

# %% [markdown]
# ## 加载音频并推理 / Load Audio and Run Inference
# %% [code]
import numpy as np
import torchaudio

print("正在加载音频...")
waveform, sr = torchaudio.load(OUTPUT_FILE)

# 重采样并转单声道
if sr != SAMPLE_RATE:
    waveform = torchaudio.functional.resample(
        waveform, orig_freq=sr, new_freq=SAMPLE_RATE
    )
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

waveform_np = waveform.squeeze().numpy()
total_samples = len(waveform_np)
total_duration = total_samples / SAMPLE_RATE
print(f"音频加载完成: 总时长 {total_duration/3600:.2f} 小时")

# 生成滑窗切片
chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
stride_samples = int(STRIDE_DURATION * SAMPLE_RATE)

chunks = []
timestamps = []

for start_idx in range(0, max(1, total_samples - chunk_samples + 1), stride_samples):
    end_idx = start_idx + chunk_samples
    chunk = waveform_np[start_idx:end_idx]
    if len(chunk) < chunk_samples:
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")
    chunks.append(chunk)
    timestamps.append((start_idx / SAMPLE_RATE, end_idx / SAMPLE_RATE))

print(f"共切分 {len(chunks)} 个 {CHUNK_DURATION} 秒片段")

# ===== 批处理推理 =====
results = []
print("开始推理...")

for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch_chunks = chunks[i : i + BATCH_SIZE]
    batch_timestamps = timestamps[i : i + BATCH_SIZE]

    for j, (audio_chunk, (start_t, end_t)) in enumerate(
        zip(batch_chunks, batch_timestamps)
    ):
        # CLAP zero-shot: 传入音频数组 + 候选标签列表
        output = audio_classifier(audio_chunk, candidate_labels=CANDIDATE_LABELS)

        # output 是 list[dict], 每个 dict 含 label / score
        # 取 top-1
        top = output[0]
        results.append(
            {
                "start": round(start_t, 2),
                "end": round(end_t, 2),
                "label": top["label"],
                "confidence": round(top["score"], 4),
            }
        )

print(f"✅ 处理完成！共 {len(results)} 条结果。")

# %% [markdown]
# ## 结果可视化 / Result Visualization
# ### 数据准备 / Data Preparation
# %% [code]
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(results)
df["start_hour"] = df["start"] / 3600.0
df["end_hour"] = df["end"] / 3600.0

labels_present = df["label"].unique()
colors = cm.get_cmap("tab20", len(labels_present))
color_map = {label: colors(i) for i, label in enumerate(labels_present)}

# 关键标签固定颜色
important_colors = {
    "Silence": "lightgray",
    "Breathing": "#2ca02c",
    "Snoring": "#ff7f0e",
    "Gasp": "#d62728",
    "Wheezing": "#9467bd",
    "Cough": "#1f77b4",
}
for k, v in important_colors.items():
    if k in color_map:
        color_map[k] = v

# %% [markdown]
# ### 1. 睡眠音频事件时间线 / Sleep Audio Event Timeline
# %% [code]
timeline_height = max(10, len(labels_present) * 0.5)
fig_tl, ax_tl = plt.subplots(figsize=(20, timeline_height))

for label in labels_present:
    subset = df[df["label"] == label]
    ax_tl.scatter(
        subset["start_hour"],
        [label] * len(subset),
        s=subset["confidence"] * 100,
        alpha=0.65,
        c=[color_map[label]],
        label=label,
        edgecolors="none",
    )

ax_tl.set_title("Sleep Audio Event Timeline", fontsize=16, fontweight="bold")
ax_tl.set_xlabel("Time (Hours)", fontsize=14)
ax_tl.set_ylabel("Sound Category", fontsize=14)
ax_tl.grid(True, axis="x", linestyle="--", alpha=0.35)
ax_tl.set_xlim(0, 8)
ax_tl.set_xticks(np.arange(0, 8.01, 0.5))
ax_tl.tick_params(axis="both", which="major", labelsize=12)
legend_max = 20
if len(labels_present) > legend_max:
    ax_tl.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, ncol=2)
else:
    ax_tl.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)
fig_tl.tight_layout()
plt.show()

# %% [markdown]
# ### 2. 事件密度热力图 / Event Density Heatmap
# %% [code]
label_list = sorted(df["label"].unique())
heatmap_height = max(10, len(label_list) * 0.5)
fig_hm, ax_hm = plt.subplots(figsize=(20, heatmap_height))
time_bins = np.arange(0, 8.5, 0.5)
heatmap_data = np.zeros((len(label_list), len(time_bins) - 1))
for i, label in enumerate(label_list):
    label_df = df[df["label"] == label]
    heatmap_data[i, :] = np.histogram(label_df["start_hour"], bins=time_bins)[0]

im = ax_hm.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax_hm.set_xticks(range(len(time_bins) - 1))
ax_hm.set_xticklabels(
    [f"{time_bins[i]:.1f}h" for i in range(len(time_bins) - 1)], fontsize=12
)
ax_hm.set_yticks(range(len(label_list)))
ax_hm.set_yticklabels(label_list, fontsize=12)
ax_hm.set_title("Event Density Heatmap (Time vs Category)", fontsize=16)
cbar = plt.colorbar(im, ax=ax_hm, label="Event Count")
cbar.ax.tick_params(labelsize=12)
ax_hm.set_xlabel("Time (Hours)", fontsize=14)
ax_hm.set_ylabel("Sound Category", fontsize=14)
fig_hm.tight_layout()
plt.show()

# %% [markdown]
# ### 3. 置信度趋势（打鼾）/ Confidence Trend (Snoring)
# %% [code]
fig_cf, ax_cf = plt.subplots(figsize=(20, 8))
target_label = "Snoring"
df_target = df[df["label"] == target_label].copy()

if not df_target.empty:
    df_target = df_target.sort_values("start").set_index("start")
    full_sec = np.arange(
        df_target.index.min(), df_target.index.max() + STRIDE_DURATION, STRIDE_DURATION
    )
    df_target = df_target.reindex(full_sec)
    df_target["confidence"] = df_target["confidence"].fillna(0)
    df_target = df_target.reset_index().rename(columns={"index": "start"})
    df_target["start_hour"] = df_target["start"] / 3600.0

    ax_cf.plot(
        df_target["start_hour"],
        df_target["confidence"],
        color=color_map.get(target_label, "orange"),
        marker=".",
        linestyle="-",
        linewidth=2,
        alpha=0.9,
        label="Original",
    )
    if len(df_target) > 3:
        window = min(7, max(3, len(df_target) // 3))
        moving_avg = df_target["confidence"].rolling(window=window, center=True).mean()
        ax_cf.plot(
            df_target["start_hour"],
            moving_avg,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Moving Avg (w={window})",
        )
    ax_cf.fill_between(
        df_target["start_hour"],
        df_target["confidence"],
        0,
        color=color_map.get(target_label, "orange"),
        alpha=0.12,
    )
    ax_cf.legend(fontsize=12)

ax_cf.set_title(f"{target_label} Confidence Trend", fontsize=16)
ax_cf.set_xlabel("Time (Hours)", fontsize=14)
ax_cf.set_ylabel("Confidence", fontsize=14)
ax_cf.grid(True, alpha=0.3)
ax_cf.set_xlim(0, 8)
ax_cf.set_xticks(np.arange(0, 8.01, 0.5))
ax_cf.set_ylim(0, 1.05)
ax_cf.tick_params(axis="both", which="major", labelsize=12)
fig_cf.tight_layout()
plt.show()

# %% [markdown]
# ### 4. 声音事件频次 / Sound Event Frequency
# %% [code]
label_counts = df["label"].value_counts()
freq_height = max(10, len(label_counts) * 0.5)
fig_fr, ax_fr = plt.subplots(figsize=(20, freq_height))
ax_fr.barh(
    label_counts.index,
    label_counts.values,
    color=[color_map.get(lbl, "gray") for lbl in label_counts.index],
)
ax_fr.set_title("Sound Event Frequency", fontsize=16)
ax_fr.set_xlabel("Count", fontsize=14)
ax_fr.grid(axis="x", alpha=0.25)
ax_fr.tick_params(axis="both", which="major", labelsize=12)
fig_fr.tight_layout()
plt.show()

# %% [markdown]
# ### 5. 各事件时间占比 / Time Duration Distribution
# %% [code]
fig_pie, ax_pie = plt.subplots(figsize=(14, 14))
duration_by_label = (df["end"] - df["start"]).groupby(df["label"]).sum()
colors_pie = [color_map.get(lbl, "gray") for lbl in duration_by_label.index]
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
# ### 6. 核心统计数据 / Core Statistics
# %% [code]
fig_st, ax_st = plt.subplots(figsize=(10, 6))
ax_st.axis("off")
stats_text = (
    f"Total Events:  {len(df)}\n"
    f"Unique Labels: {len(labels_present)}\n"
    f"Avg Confidence: {df['confidence'].mean():.3f}\n"
    f"Min Conf:       {df['confidence'].min():.3f}\n"
    f"Max Conf:       {df['confidence'].max():.3f}"
)
ax_st.text(
    0.5,
    0.5,
    stats_text,
    transform=ax_st.transAxes,
    fontsize=16,
    va="center",
    ha="center",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8, pad=1),
)
ax_st.set_title("Sleep Audio Statistics", fontsize=18, fontweight="bold")
fig_st.tight_layout()
plt.show()

# %% [markdown]
# ## 提取并播放打鼾片段 / Play Snoring Audio Clips
# %% [code]
import datetime

from IPython.display import Audio, display

snoring_events = df[df["label"] == "Snoring"].copy()
if snoring_events.empty:
    print("没有检测到打鼾 (Snoring) 事件。")
else:
    print(f"共检测到 {len(snoring_events)} 次打鼾，展示置信度最高的前 10 段：")
    top_snoring = (
        snoring_events.sort_values(by="confidence", ascending=False)
        .head(10)
        .sort_values(by="start")
    )
    try:
        for _, row in top_snoring.iterrows():
            start_t = row["start"]
            end_t   = row["end"]
            conf    = row["confidence"]
            start_str = str(datetime.timedelta(seconds=int(start_t)))
            end_str   = str(datetime.timedelta(seconds=int(end_t)))
            print("-" * 40)
            print(f"时间段: {start_str} - {end_str}  ({start_t:.2f}s -> {end_t:.2f}s) | 置信度: {conf:.4f}")
            s0 = int(start_t * SAMPLE_RATE)
            e0 = int(end_t * SAMPLE_RATE)
            seg = waveform_np[s0:e0]
            display(Audio(seg, rate=SAMPLE_RATE))
    except Exception as e:
        print(f"播放失败: {e}")
