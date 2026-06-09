import os

import cv2
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()

VIDEO_PATH = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang-small.mp4"
OUTPUT_DIR = "/kaggle/working/flux_candy_frames"
TEMP_VIDEO = "/kaggle/working/liuhuaqiang-candy-world_temp.avi"
OUTPUT_VIDEO = "/kaggle/working/liuhuaqiang-candy-world.mp4"

TEST_MODE = True
TEST_DURATION = 10
START_TIME = 0  
END_TIME = None  


os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

start_frame = int(START_TIME * fps)
if END_TIME is None:
    end_frame = total_frames
else:
    end_frame = min(int(END_TIME * fps), total_frames)

max_frames = end_frame - start_frame
print(f"Processing from {START_TIME}s to {END_TIME or 'end'}s ({max_frames} frames)")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

prompt = "keep the main subject person unchanged, transform background into candy world style, sweet colorful candy land, lollipops, candy canes, gumdrops, marshmallows, rainbow colors, whimsical fantasy background, vibrant sugary landscape, preserve foreground character"

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count >= max_frames:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(frame_rgb).convert("RGB")
    
    image = pipe(
        prompt=prompt,
        image=input_image,
        height=512,
        width=512,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device=device).manual_seed(frame_count)
    ).images[0]
    
    frame_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.png")
    image.save(frame_path)
    
    frame_count += 1
    print(f"Processed frame {frame_count}/{max_frames}")

cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs(os.path.dirname(TEMP_VIDEO), exist_ok=True)
out = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (512, 512))

for i in range(frame_count):
    frame_path = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()

print("正在使用 FFmpeg 合并音频与画面...")
ffmpeg_cmd = (
    f"ffmpeg -y "
    f"-i {TEMP_VIDEO} "
    f"-i {VIDEO_PATH} "
    f"-map 0:v:0 -map 1:a:0 "
    f"-vcodec libx264 -preset ultrafast "
    f"-c:a copy -shortest "
    f"{OUTPUT_VIDEO}"
)
os.system(ffmpeg_cmd)

os.remove(TEMP_VIDEO)
print(f"Video saved to {OUTPUT_VIDEO}")
