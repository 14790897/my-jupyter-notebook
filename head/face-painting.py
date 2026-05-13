# %% [markdown]
# # Face Painting for Video
#
# Uses **UniFace BiSeNet** to parse face regions into 19 semantic classes,
# then applies custom colors to each region for a "face painting" effect.
# Audio is preserved from the original video via ffmpeg.
#
# **Color mapping** (customizable):
# - Skin (1)       → Green  (alien skin)
# - Hair (17)       → Purple (fantasy hair)
# - Lips (12,13)    → Red    (vivid lips)
# - Eyes (4,5)      → Blue   (cyber eyes)
# - Eyebrows (2,3)  → White  (high contrast)
# - Rest             → Original image (untouched)
#
# Output: `output_painting.mp4` with original audio.
#

# %% [code]
%pip install -q "uniface[gpu]"
%pip install -q opencv-python-headless

import cv2
import numpy as np
import subprocess
from pathlib import Path

import uniface
from uniface.detection import RetinaFace
from uniface.parsing import BiSeNet
from uniface.constants import ParsingWeights

print(f"UniFace version: {uniface.__version__}")

# %% [code]
# ========== Config ==========
INPUT_VIDEO = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang-big.mp4"
OUTPUT_VIDEO = "output_painting.mp4"
CONF_THRESH = 0.5   # face detection confidence threshold
PARSER_MODEL = ParsingWeights.RESNET34  # or ParsingWeights.RESNET18
FACE_PAD_RATIO = 0.2  # padding around bbox when cropping face (ratio of bbox size)
# =============================

# Initialize models
detector = RetinaFace(confidence_threshold=CONF_THRESH)
parser = BiSeNet(model_name=PARSER_MODEL)
print("Models loaded successfully!")

# %% [code]
# ========== Color Mapping ==========
# Class IDs: 0=Background, 1=Skin, 2=Left Eyebrow, 3=Right Eyebrow,
# 4=Left Eye, 5=Right Eye, 6=Eye Glasses, 7=Left Ear, 8=Right Ear,
# 9=Ear Ring, 10=Nose, 11=Mouth, 12=Upper Lip, 13=Lower Lip,
# 14=Neck, 15=Neck Lace, 16=Cloth, 17=Hair, 18=Hat

COLOR_MAP = {
    1:  (0, 255, 0),     # Skin       → Green (BGR)
    2:  (255, 255, 255), # Left Eyebrow  → White
    3:  (255, 255, 255), # Right Eyebrow → White
    4:  (255, 0, 0),     # Left Eye    → Blue
    5:  (255, 0, 0),     # Right Eye   → Blue
    10: (0, 255, 255),   # Nose        → Yellow
    12: (0, 0, 255),     # Upper Lip   → Red
    13: (0, 0, 255),     # Lower Lip   → Red
    17: (255, 0, 255),   # Hair        → Purple
}
# Classes NOT in COLOR_MAP will keep original image colors.
# ====================================

# %% [code]
# Open source video, get parameters
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info: {w}x{h}, {fps:.1f} FPS, total {total_frames} frames")

# ffmpeg command:
#   input 0: stdin video frames (processed)
#   input 1: original video file (for audio)
command = [
    "ffmpeg",
    "-y",
    "-loglevel", "error",
    # input 0: video frames (pipe)
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-s", f"{w}x{h}",
    "-pix_fmt", "bgr24",
    "-r", str(fps),
    "-i", "-",
    # input 1: original video (audio source)
    "-i", INPUT_VIDEO,
    # encoding
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    "-preset", "fast",
    "-c:a", "copy",
    "-map", "0:v",
    "-map", "1:a",
    OUTPUT_VIDEO,
]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
print(f"Output video (ffmpeg): {OUTPUT_VIDEO}")
print(f"Audio will be copied from original video.")

# %% [code]
def crop_face_from_frame(frame, bbox, pad_ratio=0.2):
    """Crop face region from frame with padding.

    bbox: [x1, y1, x2, y2]
    Returns: cropped image, and the crop coordinates (x1, y1, x2, y2) in original frame.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Add padding
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)

    cx1 = max(0, x1 - pw)
    cy1 = max(0, y1 - ph)
    cx2 = min(w_frame, x2 + pw)
    cy2 = min(h_frame, y2 + ph)

    crop = frame[cy1:cy2, cx1:cx2]
    return crop, (cx1, cy1, cx2, cy2)


def apply_face_painting(crop, mask, color_map):
    """Apply color mapping to a face crop based on parsing mask.

    crop: BGR image (face region)
    mask: int array, same HxW as crop, values in [0..18]
    color_map: dict {class_id: (B, G, R)}
    """
    result = crop.copy()

    for class_id, color in color_map.items():
        region_mask = (mask == class_id)
        if np.any(region_mask):
            result[region_mask] = color

    return result


# Processing loop
results_log = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    faces = detector.detect(frame)

    face_count = 0
    for face in faces:
        bbox = face.bbox  # [x1, y1, x2, y2]
        conf = face.confidence if hasattr(face, 'confidence') else 0

        if conf < CONF_THRESH:
            continue

        face_count += 1

        # Crop face region with padding
        crop, (cx1, cy1, cx2, cy2) = crop_face_from_frame(
            frame, bbox, pad_ratio=FACE_PAD_RATIO
        )

        if crop.size == 0:
            continue

        # Parse face (BiSeNet expects a face crop)
        mask = parser.parse(crop)  # mask shape = crop HxW, values in [0..18]

        # Apply colors to the crop
        painted_crop = apply_face_painting(crop, mask, COLOR_MAP)

        # Paste back to frame
        annotated[cy1:cy2, cx1:cx2] = painted_crop

    # Overlay text
    cv2.putText(annotated, "Face Painting", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(annotated, f"Face count: {face_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    timestamp = frame_idx / fps
    cv2.putText(annotated, f"Frame: {frame_idx}  Time: {timestamp:.1f}s",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    results_log.append({
        "frame": frame_idx,
        "time_s": round(timestamp, 2),
        "face_count": face_count,
    })

    # Write frame via ffmpeg pipe
    proc.stdin.write(annotated.tobytes())
    frame_idx += 1

    if frame_idx % 100 == 0:
        pct = frame_idx / max(total_frames, 1) * 100
        print(f"Progress: {frame_idx}/{total_frames} ({pct:.1f}%)")

# Cleanup
proc.stdin.close()
proc.wait()
cap.release()
print(f"\nDone! Total {frame_idx} frames")
print(f"Output saved to: {OUTPUT_VIDEO}")

# %% [code]
# Summary
print("=" * 50)
print("Face Painting Summary:")
total_with_face = sum(1 for e in results_log if e["face_count"] > 0)
print(f"  Total frames with face detected: {total_with_face} / {len(results_log)}")
print(f"  Colored classes: {list(COLOR_MAP.keys())}")
print("=" * 50)

# %% [code]
# Optional: save log as CSV
import pandas as pd

df = pd.DataFrame(results_log)
csv_path = OUTPUT_VIDEO.replace(".mp4", "_log.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Log saved to: {csv_path}")
df.head(10)
