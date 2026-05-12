# %% [markdown]
# # Face Mosaic / Blur for Video
#
# Uses **UniFace XSeg** to segment face regions in each frame,
# then applies mosaic or blur to anonymize faces.
# Audio is preserved from the original video via ffmpeg.
#
# **Two processing modes**:
# - `MODE = "mosaic"` — pixelated face
# - `MODE = "blur"`   — Gaussian blur on face region
#
# Output: `output_mosaic.mp4` with original audio.
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
from uniface.parsing import XSeg

print(f"UniFace version: {uniface.__version__}")

# %% [code]
# ========== Config ==========
INPUT_VIDEO = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang.mp4"
OUTPUT_VIDEO = "output_mosaic.mp4"
MODE = "mosaic"          # "mosaic" or "blur"
MOSAIC_BLOCK = 15        # mosaic block size (pixels), smaller = heavier mosaic
BLUR_KERNEL = (99, 99) # Gaussian blur kernel size (must be odd)
CONF_THRESH = 0.5        # face detection confidence threshold
# =============================

# Initialize models
detector = RetinaFace(confidence_threshold=CONF_THRESH)
xseg = XSeg(blur_sigma=3)   # light smoothing on the mask
print("Models loaded successfully!")

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
    "-c:a", "copy",   # copy original audio stream directly
    "-map", "0:v",
    "-map", "1:a",
    OUTPUT_VIDEO,
]
proc = subprocess.Popen(command, stdin=subprocess.PIPE)
print(f"Output video (ffmpeg): {OUTPUT_VIDEO}")
print(f"Audio will be copied from original video.")
print(f"Mode: {MODE}")

# %% [code]
def apply_mosaic(region, block_size=15):
    """Apply mosaic effect to a region."""
    bh, bw = region.shape[:2]
    # shrink
    small_w = max(1, bw // block_size)
    small_h = max(1, bh // block_size)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    # enlarge back
    mosaic = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
    return mosaic


def apply_face_mask(frame, mask, mode="mosaic", block_size=15, blur_kernel=(99, 99)):
    """Apply mosaic or blur to face region defined by mask.

    mask: [0, 1] float, same HxW as frame
    """
    frame_f = frame.astype(np.float32)
    mask_3ch = np.stack([mask] * 3, axis=-1).astype(np.float32)

    if mode == "mosaic":
        mosaic_region = apply_mosaic(frame, block_size).astype(np.float32)
        result = frame_f * (1 - mask_3ch) + mosaic_region * mask_3ch
    else:  # blur
        blurred = cv2.GaussianBlur(frame, blur_kernel, 0).astype(np.float32)
        result = frame_f * (1 - mask_3ch) + blurred * mask_3ch

    return result.clip(0, 255).astype(np.uint8)

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

    if len(faces) > 0:
        # Process the largest face (closest to camera)
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        face = faces_sorted[0]

        if face.landmarks is not None:
            # XSeg requires landmarks for alignment
            mask = xseg.parse(frame, landmarks=face.landmarks)

            # Ensure mask is same size as frame
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Apply face anonymization
            annotated = apply_face_mask(
                annotated,
                mask,
                mode=MODE,
                block_size=MOSAIC_BLOCK,
                blur_kernel=BLUR_KERNEL,
            )
            face_count = 1

            # Draw green contour of face mask (optional visualization)
            mask_u8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

    # Overlay text
    mode_label = "Mosaic" if MODE == "mosaic" else "Blur"
    cv2.putText(annotated, f"Face {mode_label}", (20, 40),
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
print("Face Detection Summary:")
total_with_face = sum(1 for e in results_log if e["face_count"] > 0)
print(f"  Total frames with face detected: {total_with_face} / {len(results_log)}")
print(f"  Face anonymization mode: {MODE}")
print("=" * 50)

# %% [code]
# Optional: save log as CSV
import pandas as pd

df = pd.DataFrame(results_log)
csv_path = OUTPUT_VIDEO.replace(".mp4", "_log.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Log saved to: {csv_path}")
df.head(10)
