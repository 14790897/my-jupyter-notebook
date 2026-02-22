# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-21T09:00:56.836011Z","iopub.execute_input":"2026-02-21T09:00:56.836785Z","iopub.status.idle":"2026-02-21T09:01:00.429820Z","shell.execute_reply.started":"2026-02-21T09:00:56.836753Z","shell.execute_reply":"2026-02-21T09:01:00.428808Z"}}
# ================= 环境准备：安装依赖 =================
# 安装包含 solutions API 的稳定版本
!pip install mediapipe opencv-python -q

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-21T09:01:00.439442Z","iopub.status.idle":"2026-02-21T09:01:00.439799Z","shell.execute_reply.started":"2026-02-21T09:01:00.439605Z","shell.execute_reply":"2026-02-21T09:01:00.439627Z"}}
import torch
import random
import os
import numpy as np
from IPython.display import Image, Video
import cv2
import mediapipe as mp

# ================= 1. 基础配置与环境设置 =================
SEED = 42
DEVICE_ID = 0          

INPUT_VIDEO_PATH = '/kaggle/input/datasets/liuweiq/daxiaonailong/caixunkun.mp4'
INPUT_BASENAME = os.path.splitext(os.path.basename(INPUT_VIDEO_PATH))[0]
OUTPUT_DIR = '/kaggle/working/runs/pose/predict'
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, f'{INPUT_BASENAME}_black_mesh.mp4')
FINAL_MP4_PATH = f'compressed_{INPUT_BASENAME}.mp4'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📂 输出目录已就绪: {OUTPUT_DIR}")

# 推理参数（当前未直接使用，保留供扩展）
CONF_THRESHOLD = 0.1    
IOU_THRESHOLD = 0.6      
VID_STRIDE = 1

# %% [code] {"execution":{"iopub.status.busy":"2026-02-21T09:01:00.440651Z","iopub.status.idle":"2026-02-21T09:01:00.440894Z","shell.execute_reply.started":"2026-02-21T09:01:00.440782Z","shell.execute_reply":"2026-02-21T09:01:00.440797Z"},"jupyter":{"outputs_hidden":false}}
# ================= 固定随机种子以保证可复现性 =================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-21T09:01:00.443855Z","iopub.status.idle":"2026-02-21T09:01:00.444169Z","shell.execute_reply.started":"2026-02-21T09:01:00.444046Z","shell.execute_reply":"2026-02-21T09:01:00.444061Z"}}
# ================= 2. 初始化MediaPipe Holistic（按照官方文档格式） =================
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# ================= 3. 视频读写初始化与处理 =================
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

print(f"🎬 开始使用 MediaPipe Holistic 提取全身【姿态+面部+手势】网格...")

# ================= 4. 使用 with 语句处理视频流（官方推荐方式）=================
frame_count = 0
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("视频处理完成")
            break
            
        frame_count += 1
        
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # 创建纯黑画布用于绘制骨架
        black_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw face landmarks
        mp_drawing.draw_landmarks(
            black_canvas,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        
        mp_drawing.draw_landmarks(
            black_canvas,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            black_canvas,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())

        # Draw left hand landmarks
        mp_drawing.draw_landmarks(
            black_canvas,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Draw right hand landmarks
        mp_drawing.draw_landmarks(
            black_canvas,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        out.write(black_canvas)
        print(f"处理进度: 第 {frame_count} 帧", end='\r')

cap.release()
out.release()
print(f"\n✅ MediaPipe Holistic 全身姿态视频生成完成！已保存至 {OUTPUT_VIDEO_PATH}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-21T09:01:00.445275Z","iopub.status.idle":"2026-02-21T09:01:00.445716Z","shell.execute_reply.started":"2026-02-21T09:01:00.445437Z","shell.execute_reply":"2026-02-21T09:01:00.445459Z"}}
# ================= 5. FFmpeg 音视频合并与压缩 =================
print("正在合并音频与画面...")
ffmpeg_cmd = (
    f'ffmpeg -y '
    f'-i {OUTPUT_VIDEO_PATH} '         # 视频源：姿态检测生成的画面
    f'-i {INPUT_VIDEO_PATH} '          # 音频源：原始输入视频
    f'-map 0:v:0 -map 1:a:0 '
    f'-vcodec libx264 -preset ultrafast -vf scale=1080:-2 '
    f'-c:a copy -shortest '
    f'{FINAL_MP4_PATH}'
)
os.system(ffmpeg_cmd)
print(f"✅ 合并压缩完成！最终文件: {FINAL_MP4_PATH}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-21T09:01:00.447019Z","iopub.status.idle":"2026-02-21T09:01:00.447347Z","shell.execute_reply.started":"2026-02-21T09:01:00.447181Z","shell.execute_reply":"2026-02-21T09:01:00.447196Z"}}
# ================= 6. 在Notebook中展示最终视频 =================
display(Video(FINAL_MP4_PATH, embed=True, width=640))