# %% [markdown]
# # Sleep Posture Detection from Video (Side-View Overhead Camera)
# **场景**：摄像头安装在人体右侧，俯视拍摄（右侧俯视）。
# 目标：逐帧分析头部姿态，判断是否为侧睡，并检测睡姿变化。
# **判定逻辑**：
# - `|roll| < 30°` → 仰睡（平躺）
# - `|roll| ≥ 30°` → 侧睡
# - 连续帧之间睡姿状态发生变化 → 标记为「睡姿改变」并记录时间戳
# 输出：带标注的视频文件（睡姿标签 + 变化提示）

# %% [code]
%pip install -q "uniface[gpu]"
%pip install -q opencv-python-headless

import cv2
import numpy as np
import subprocess
from pathlib import Path
import uniface
from uniface.detection import RetinaFace
from uniface.headpose import HeadPose
from uniface.draw import draw_head_pose

print(f"UniFace version: {uniface.__version__}")

# %% [code]
# ========== 配置区 ==========
INPUT_VIDEO = "/kaggle/input/datasets/liuweiq/my-sleep-record/00-00-60-allday.mp4"      # 输入视频路径，请修改为实际路径
OUTPUT_VIDEO = "output_annotated.mp4"  # 输出标注视频路径
ROLL_THRESHOLD = 30.0                # 侧睡判定阈值（度），|roll| >= 此值视为侧睡
MIN_FACE_AREA_RATIO = 0.0001        # 最小人脸面积占帧比例，过滤误检
DISPLAY_DURATION = 60                # 睡姿改变提示持续帧数
# =============================

# 初始化模型
detector = RetinaFace(confidence_threshold=0.5)
head_pose = HeadPose()
print("Models loaded successfully!")

# %% [code]
# 打开视频
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_area = w * h

print(f"Video info: {w}x{h}, {fps:.1f} FPS, total {total_frames} frames")

# 使用 ffmpeg 管道写入视频（比 cv2.VideoWriter 更稳定，支持 H.264）
command = [
    "ffmpeg",
    "-y",                      # 覆盖已存在文件
    "-loglevel", "error",      # 只输出错误信息
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-s", f"{w}x{h}",
    "-pix_fmt", "bgr24",
    "-r", str(fps),
    "-i", "-",                 # 从 stdin 读取帧数据
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    "-preset", "fast",
    OUTPUT_VIDEO,
]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
print(f"Output video (ffmpeg): {OUTPUT_VIDEO}")

# %% [code]
# 状态变量
prev_posture = None        # Previous posture: "Supine" / "Side" / None
posture_changed = False    # 当前帧是否发生睡姿改变
change_display_counter = 0 # 提示剩余显示帧数
results_log = []           # 每帧结果日志

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    faces = detector.detect(frame)

    # 默认标签（无人脸时沿用上一帧）
    posture_label = prev_posture if prev_posture else "Unknown"
    yaw = pitch = roll = None

    # 选最大的人脸（最接近相机）
    if len(faces) > 0:
        faces_sorted = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        face = faces_sorted[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        # 过滤过小检测框
        face_area = (x2 - x1) * (y2 - y1)
        if face_area / frame_area >= MIN_FACE_AREA_RATIO:
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                result = head_pose.estimate(face_crop)
                yaw, pitch, roll = result.yaw, result.pitch, result.roll

                # 绘制头部姿态立方体
                draw_head_pose(annotated, face.bbox, pitch, yaw, roll)

                # 根据 roll 角判断睡姿（右侧俯视视角）
                # roll ≈ 0  → 仰睡（平躺，头部基本水平）
                # |roll|大   → 侧睡（头部侧倾）
                if abs(roll) < ROLL_THRESHOLD:
                    posture_label = "Supine"
                else:
                    posture_label = "Side"

                # 画人脸框（可选，注释掉以移除）
                # cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 显示角度数值
                angle_text = f"yaw:{yaw:.1f} pitch:{pitch:.1f} roll:{roll:.1f}"
                cv2.putText(annotated, angle_text, (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---- 睡姿变化检测 ----
    if prev_posture is not None and posture_label != "Unknown" and posture_label != prev_posture:
        posture_changed = True
        change_display_counter = DISPLAY_DURATION
        print(f"[Frame {frame_idx}] Posture Changed: {prev_posture} → {posture_label}  (time={frame_idx/fps:.1f}s)")

    if posture_changed:
        change_display_counter -= 1
        if change_display_counter <= 0:
            posture_changed = False

    prev_posture = posture_label if posture_label != "Unknown" else prev_posture

    # ---- 绘制标注 ----
    # 睡姿标签（左上角）
    color = (0, 255, 0) if posture_label == "Supine" else (0, 0, 255) if posture_label == "Side" else (128, 128, 128)
    label_text = f"Posture: {posture_label}"
    cv2.putText(annotated, label_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # 睡姿改变闪烁提示
    if posture_changed:
        # 每隔 5 帧闪烁红色警告
        if (change_display_counter // 5) % 2 == 0:
            cv2.putText(annotated, "*** Posture Changed ***", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # 帧号 & 时间戳
    timestamp = frame_idx / fps
    info_text = f"Frame: {frame_idx}  Time: {timestamp:.1f}s"
    cv2.putText(annotated, info_text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 记录日志
    results_log.append({
        "frame": frame_idx,
        "time_s": round(timestamp, 2),
        "yaw": round(yaw, 2) if yaw is not None else None,
        "pitch": round(pitch, 2) if pitch is not None else None,
        "roll": round(roll, 2) if roll is not None else None,
        "posture": posture_label,
    })

    # 通过 ffmpeg 管道写入帧
    proc.stdin.write(annotated.tobytes())
    frame_idx += 1

    # 进度打印（每 100 帧）
    if frame_idx % 100 == 0:
        print(f"Progress: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

# 清理 ffmpeg 进程
proc.stdin.close()
proc.wait()
cap.release()
print(f"\nDone! Total {frame_idx} frames")
print(f"Annotated video saved to: {OUTPUT_VIDEO}")

# %% [code]
# 汇总睡姿变化时刻
print("=" * 50)
print("Posture Change Summary:")
change_events = []
for i, entry in enumerate(results_log):
    if i == 0:
        continue
    if entry["posture"] != "Unknown" and results_log[i-1]["posture"] != "Unknown":
        if entry["posture"] != results_log[i-1]["posture"]:
            change_events.append(entry)
            print(f"  {entry['time_s']:.1f}s (Frame {entry['frame']}): "
                  f"{results_log[i-1]['posture']} → {entry['posture']}  "
                  f"roll={entry['roll']:.1f}°")

print(f"Total posture changes detected: {len(change_events)}")
print("=" * 50)

# %% [code]
# 可选：保存日志为 CSV
import pandas as pd

df = pd.DataFrame(results_log)
csv_path = OUTPUT_VIDEO.replace(".mp4", "_log.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Log saved to: {csv_path}")
df.head(10)
