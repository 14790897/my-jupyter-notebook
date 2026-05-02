# %% [markdown]
# ## 使用 AST 模型进行音频分类 简单演示
# %% [code]
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
# %% [markdown]
# ## 转换 M4A 文件为 WAV 格式
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
# %% [markdown]
# ## 加载睡眠音频，正式评估打鼾
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
# ===== 2. 拆分为两张图：主仪表板与统计面板 =====

# ---- 主仪表板：时间线、热力图、置信度趋势纵向排列 ----
fig1 = plt.figure(figsize=(18, 14))
gs1 = fig1.add_gridspec(3, 1, hspace=0.38)
ax_timeline = fig1.add_subplot(gs1[0, 0])
ax_heatmap = fig1.add_subplot(gs1[1, 0])
ax_confidence = fig1.add_subplot(gs1[2, 0])

# 时间线
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

ax_timeline.set_title("🕐 Sleep Audio Event Timeline", fontsize=14, fontweight='bold')
ax_timeline.set_xlabel("Time (Hours)")
ax_timeline.set_ylabel("Sound Category")
ax_timeline.grid(True, axis='x', linestyle='--', alpha=0.35)
ax_timeline.set_xlim(0, 8)
ax_timeline.set_xticks(np.arange(0, 8.01, 0.5))
# 如果标签过多，缩小图例并换列显示
legend_max = 20
if len(labels_present) > legend_max:
    ax_timeline.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)
else:
    ax_timeline.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

# 热力图数据准备
time_bins = np.arange(0, 8.5, 0.5)
label_list = sorted(df['label'].unique())
heatmap_data = np.zeros((len(label_list), len(time_bins) - 1))
for i, label in enumerate(label_list):
    label_df = df[df['label'] == label]
    heatmap_data[i, :] = np.histogram(label_df['start_hour'], bins=time_bins)[0]

im = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax_heatmap.set_xticks(range(len(time_bins) - 1))
ax_heatmap.set_xticklabels([f"{time_bins[i]:.1f}h" for i in range(len(time_bins) - 1)], fontsize=9)
ax_heatmap.set_yticks(range(len(label_list)))
ax_heatmap.set_yticklabels(label_list, fontsize=9)
ax_heatmap.set_title("🔥 Event Density Heatmap (Time vs Category)")
plt.colorbar(im, ax=ax_heatmap, label="Event Count")
ax_heatmap.set_xlabel("Time (Hours)")
ax_heatmap.set_ylabel("Sound Category")

# 置信度趋势（目标标签）
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
    ax_confidence.legend(fontsize=9)

ax_confidence.set_title(f"📈 {target_label} Confidence Trend")
ax_confidence.set_xlabel("Time (Hours)")
ax_confidence.set_ylabel("Confidence")
ax_confidence.grid(True, alpha=0.3)
ax_confidence.set_xlim(0, 8)
ax_confidence.set_xticks(np.arange(0, 8.01, 0.5))
ax_confidence.set_ylim(0, 1.05)

fig1.suptitle("🌙 Sleep Audio Overview", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# ---- 统计面板：频次柱状图、饼图、统计文本纵向排列 ----
fig2 = plt.figure(figsize=(12, 14))
gs2 = fig2.add_gridspec(3, 1, hspace=0.38)
ax_freq = fig2.add_subplot(gs2[0, 0])
ax_pie = fig2.add_subplot(gs2[1, 0])
ax_stats = fig2.add_subplot(gs2[2, 0])

# 频次柱状图
label_counts = df['label'].value_counts()
ax_freq.barh(label_counts.index, label_counts.values, color=[color_map.get(label_name, 'gray') for label_name in label_counts.index])
ax_freq.set_title('📊 Sound Event Frequency')
ax_freq.set_xlabel('Count')
ax_freq.grid(axis='x', alpha=0.25)

# 饼图（时间占比）
duration_by_label = (df['end'] - df['start']).groupby(df['label']).sum()
colors_pie = [color_map.get(label, 'gray') for label in duration_by_label.index]
ax_pie.pie(duration_by_label.values, labels=list(duration_by_label.index), autopct='%1.1f%%',
           colors=colors_pie, startangle=90, textprops={'fontsize': 9})
ax_pie.set_title('🥧 Time Duration Distribution')

# 统计文本
ax_stats.axis('off')
stats_text = f"Total Events: {len(df)}\nUnique Labels: {len(labels_present)}\nAvg Confidence: {df['confidence'].mean():.3f}\nMin Conf: {df['confidence'].min():.3f}\nMax Conf: {df['confidence'].max():.3f}"
ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes, fontsize=11, va='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.6))

fig2.suptitle('🧾 Sleep Audio Statistics', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
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
