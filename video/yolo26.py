# %% [markdown]
# # 视频语义分割任务 - 使用 YOLO26 SemanticMask
# ## https://docs.ultralytics.com/modes/predict#semanticmask
# %% [code]
%pip install -U ultralytics -q

# %% [markdown]
# ## 模型推理参数配置

# %% [code]
INPUT_VIDEO_PATH = '/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang-small.mp4'
CONF_THRESHOLD = 0.1    # 置信度
VID_STRIDE = 1          # 抽帧率 (1为不跳帧)

# %% [markdown]
# ## 语义分割推理

# %% [code]
import torch
import random
import os
import numpy as np
import cv2
from ultralytics import YOLO
from IPython.display import Image, Video, clear_output

SEED = 42
DEVICE_ID = 0

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

MODEL_FILENAME = 'yolo26x-sem.pt'
model = YOLO(MODEL_FILENAME)

INPUT_BASENAME = os.path.splitext(os.path.basename(INPUT_VIDEO_PATH))[0]
YOLO_OUTPUT_AVI = f'/kaggle/working/runs/segment/predict/{INPUT_BASENAME}_semantic.avi'
FINAL_MP4_PATH = f'compressed_{INPUT_BASENAME}_semantic.mp4'

print(f"开始对 {INPUT_VIDEO_PATH} 进行语义分割推理...")
results = model.predict(
    source=INPUT_VIDEO_PATH, 
    conf=CONF_THRESHOLD,
    save=False,          
    device=DEVICE_ID,            
    stream=True,        
    vid_stride=VID_STRIDE,
)

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

out_fps = fps
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

os.makedirs(os.path.dirname(YOLO_OUTPUT_AVI), exist_ok=True)
writer = cv2.VideoWriter(YOLO_OUTPUT_AVI, fourcc, out_fps, (w, h))

frame_count = 0
for r in results:
    semantic_mask = r.semantic_mask
    
    if semantic_mask is not None:
        class_ids = semantic_mask.data.unique()
        print(f"Frame {frame_count}: 检测到 {len(class_ids)} 类物体")
        for class_id in class_ids:
            class_name = model.names.get(int(class_id), f"class_{class_id}")
            print(f"  - {class_name} (ID: {class_id})")
    
    frame = r.plot()
    writer.write(frame)
    frame_count += 1
    if frame_count % 30 == 0:
        clear_output(wait=True)
        print(f"已处理 {frame_count} 帧...")

writer.release()
print(f"✅ 推理完成！共处理 {frame_count} 帧。输出：{YOLO_OUTPUT_AVI}")

# %% [code]
print("正在合并音频与画面...")
ffmpeg_cmd = (
    f'ffmpeg -y '
    f'-i {YOLO_OUTPUT_AVI} '
    f'-i {INPUT_VIDEO_PATH} '
    f'-map 0:v:0 -map 1:a:0 '
    f'-vcodec libx264 -preset ultrafast -vf scale=1080:-2 '
    f'-c:a copy -shortest '
    f'{FINAL_MP4_PATH}'
)
os.system(ffmpeg_cmd)
print(f"✅ 合并压缩完成！最终文件: {FINAL_MP4_PATH}")

# %% [markdown]
# ## 展示结果

# %% [code]
display(Video(FINAL_MP4_PATH, embed=True, width=640))
