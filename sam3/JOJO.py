# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # JOJO vs DIO zero shot 识别任务

# %% [code] {"jupyter":{"outputs_hidden":false}}
%pip install -U ultralytics -q  # SAM 3 需要 ultralytics >= 8.3.237
# ⚠️ sam3.pt 权重需从 HuggingFace 手动申请并下载: https://huggingface.co/models
# 下载后将 sam3.pt 放置于工作目录，或在 MODEL_FILENAME 中指定完整路径
# 若出现 TypeError: 'SimpleTokenizer' object is not callable，执行:
# pip uninstall clip -y && pip install git+https://github.com/ultralytics/CLIP.git

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 模型推理参数是着重需要配置的地方

# %% [code] {"jupyter":{"outputs_hidden":false}}
INPUT_VIDEO_PATH = '/kaggle/input/datasets/liuweiq/daxiaonailong/JOJO vs DIO.mp4'
CONF_THRESHOLD = 0.1    # 置信度 (卡通形象建议调低至 0.05 - 0.1)
IMG_SIZE = 640           # 推理分辨率
VID_STRIDE = 1           # 抽帧率 (1为不跳帧)
# 目标检测设置 (Open-Vocabulary 文本提示词，使用简单名词短语)
TARGET_CLASSES = [
    "anime man with black Coat",
    "anime man with Yellow Coat"
]
# 易读名字 (对应 TARGET_CLASSES 顺序)
SHORT_LABELS = ["JOJO", "DIO"]

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 推理

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
import random
import os
import numpy as np
from IPython.display import Video
from ultralytics.models.sam import SAM3VideoSemanticPredictor

# 1. 基础环境设置
SEED = 42
DEVICE_ID = 0          # GPU 设备号

 # 2. 模型设置 (需手动从 HuggingFace 下载 sam3.pt)
MODEL_FILENAME = '/kaggle/input/models/liuweiq/sam3-fb/pytorch/default/1/sam3.pt'

# 3. 路径与文件设置
INPUT_BASENAME = os.path.splitext(os.path.basename(INPUT_VIDEO_PATH))[0]
OUTPUT_AVI = f'/kaggle/working/{INPUT_BASENAME}_sam3.avi'
FINAL_MP4_PATH = f'compressed_{INPUT_BASENAME}.mp4'

# %% [code] {"jupyter":{"outputs_hidden":false}}
import cv2
from IPython.display import clear_output

# 第一步：固定随机种子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# 第二步：初始化 SAM3VideoSemanticPredictor
# SAM 3 使用文本概念提示词，自动检测并跟踪视频中所有匹配实例
overrides = dict(
    conf=CONF_THRESHOLD,
    task="segment",
    mode="predict",
    imgsz=IMG_SIZE,
    model=MODEL_FILENAME,
    half=True,   # FP16 加速
    save=False,
    device=DEVICE_ID,
    vid_stride=VID_STRIDE,
)
predictor = SAM3VideoSemanticPredictor(overrides=overrides)

# 第三步：开始流式推理
print(f"开始对 {INPUT_VIDEO_PATH} 进行 SAM 3 推理...")
print(f"文本提示词: {TARGET_CLASSES}")
results = predictor(
    source=INPUT_VIDEO_PATH,
    text=TARGET_CLASSES,
    stream=True,
)

# 用原视频读取宽高/帧率
os.makedirs(os.path.dirname(OUTPUT_AVI), exist_ok=True)
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

out_fps = fps
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_AVI, fourcc, out_fps, (w, h))

frame_count = 0
for r in results:
    frame = r.plot()  # BGR，SAM 3 会在画面上标注概念标签与分割掩码
    writer.write(frame)
    frame_count += 1
    if frame_count % 30 == 0:
        clear_output(wait=True)
        print(f"已处理 {frame_count} 帧...")

writer.release()
print(f"✅ 推理完成！共处理 {frame_count} 帧。输出：{OUTPUT_AVI}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# 第五步：FFmpeg 音视频合并压缩

import subprocess
print("正在合并音频与画面...")
# 将命令和参数严格拆分为列表元素
ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-i', OUTPUT_AVI,               # 视频源：SAM 3 生成的画面
    '-i', INPUT_VIDEO_PATH,         # 音频源：同源输入视频
    '-map', '0:v:0', 
    '-map', '1:a:0',
    '-vcodec', 'libx264', 
    '-preset', 'ultrafast', 
    '-vf', 'scale=1080:-2',
    '-c:a', 'copy', 
    '-shortest',
    FINAL_MP4_PATH                  # 输出路径：变量直接放进来，绝对不会被空格截断
]

try:
    # 执行命令，check=True 表示如果 FFmpeg 报错会直接抛出 Python 异常
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"✅ 合并压缩完成！最终文件: {FINAL_MP4_PATH}")
except subprocess.CalledProcessError as e:
    print(f"❌ 合并失败，FFmpeg 返回错误码: {e.returncode}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 展示 (定位跳转)

# %% [code] {"jupyter":{"outputs_hidden":false}}
display(Video(FINAL_MP4_PATH, embed=True, width=640))