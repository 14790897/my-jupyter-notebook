# %% [code]
# 1. 安装依赖
from IPython.display import clear_output
!pip install emotiefflib[all] ultralytics moviepy
!apt install -y fonts-wqy-zenhei fonts-noto-cjk
clear_output()
print("✅ 依赖安装完成")

# %% [code]
# 2. 导入依赖库
import os
import cv2
import numpy as np
import urllib.request
import subprocess
import shutil
import time
from ultralytics import YOLO
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from PIL import Image, ImageDraw, ImageFont

print("✅ 库导入完成")

# %% [code]
# 3. 下载人脸检测模型
def ensure_face_model(model_path="yolov8n-face.pt"):
    """检查并自动下载 YOLOv8 人脸模型"""
    if os.path.exists(model_path):
        print(f"✅ 模型已存在: {model_path}")
        return True
        
    print(f"未找到 {model_path}，正在自动下载...")
    url = "https://huggingface.co/deepghs/yolo-face/resolve/739664f2d00e436a8882238f83175ab0f6497578/yolov8n-face/model.pt"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("✅ 模型下载成功！")
        return True
    except Exception as e:
        print(f"❌ 自动下载失败: {e}")
        return False

# 运行：下载模型
ensure_face_model("yolov8n-face.pt")

# %% [code]
# 4. 查找中文字体路径
def find_chinese_font():
    """查找系统中可用的中文字体"""
    font_paths = [
        # Kaggle常见路径
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
        "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto/NotoSansCJK-Bold.ttc",
        # 本地Windows路径
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        # 当前目录
        "simhei.ttf",
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            print(f"✅ 找到字体: {path}")
            return path
    
    print("⚠️ 未找到中文字体，将使用默认字体")
    return None

# 运行：查找字体
FONT_PATH = find_chinese_font()

# %% [code]
# 5. 加载模型（全局变量）
print("正在加载模型...")

# 加载人脸检测模型
face_detector = YOLO("yolov8n-face.pt")
print("✅ YOLOv8 人脸检测模型加载完成")

# 加载EmotiEffLib模型
fer = EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_best_vgaf", device="cuda")
print("✅ EmotiEffLib 模型加载完成")

# 加载中文字体
if FONT_PATH:
    chinese_font = ImageFont.truetype(FONT_PATH, 30)
    print(f"✅ 中文字体加载完成: {FONT_PATH}")
else:
    chinese_font = ImageFont.load_default()
    print("⚠️ 使用默认字体")

# %% [code]
# 6. 定义并运行：提取人脸并预测专注度
def extract_faces_and_predict(video_path, window_size=128):
    """
    第一趟：读取视频，提取人脸，预测专注度
    返回：frame_status_map, frame_emotion_map, video_info
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息：{total_frames} 帧, FPS: {fps:.2f}, 分辨率: {w}x{h}")

    face_data_list = []
    frame_count = 0

    # 第一趟：提取人脸
    print("开始提取人脸...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = face_detector(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) > 0:
            # 取面积最大的脸
            if len(boxes) > 1:
                boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            
            x1, y1, x2, y2 = map(int, boxes[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_bgr = frame[y1:y2, x1:x2]
            if face_bgr.shape[0] >= 20 and face_bgr.shape[1] >= 20:
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_data_list.append({
                    "frame_num": frame_count,
                    "box": (x1, y1, x2, y2),
                    "face_rgb": face_rgb
                })

        if frame_count % 200 == 0:
            print(f"提取进度: {frame_count}/{total_frames} 帧")

    cap.release()
    print(f"✅ 人脸提取完成，共 {len(face_data_list)} 帧有效人脸")

    if len(face_data_list) == 0:
        print("❌ 未检测到人脸")
        return None

    all_faces = [data["face_rgb"] for data in face_data_list]
    n = len(all_faces)

    # -------------------------------------------------------
    # 正确做法：直接使用 predict_engagement
    # 官方说明：predict_engagement 内部自动完成特征提取 + 滑动窗口
    # 输入：人脸图像序列
    # 输出：(labels, scores)，labels 长度为 n - window_size（滑动窗口数）
    # -------------------------------------------------------
    print(f"正在预测专注度（共 {n} 帧）...")
    engagement_labels, _ = fer.predict_engagement(all_faces, sliding_window_width=window_size)
    # ⚠️ 关键细节：engagement_labels 长度为 n - window_size，不是 n！
    # engagement_labels[i] 对应 face_data_list[i + window_size - 1]
    # 即：窗口 0（帧[0..127]）的预测归属给第 127 帧
    eng_count = len(engagement_labels)
    print(f"专注度预测：输入 {n} 帧 → 输出 {eng_count} 个预测（滑动窗口 = {window_size} 帧）")

    # -------------------------------------------------------
    # 情绪：逐帧独立预测
    # -------------------------------------------------------
    print(f"正在提取情绪特征（共 {n} 帧）...")
    features = fer.extract_features(all_faces)
    _, emotion_scores = fer.classify_emotions(features, logits=True)

    # ===== 调试：打印情绪模型输出信息 =====
    global_emo_mean = np.mean(emotion_scores, axis=0)
    global_emo_idx  = np.argmax(global_emo_mean)
    print("\n" + "="*50)
    print("🔍 调试信息：情绪模型全局输出")
    print("="*50)
    print(f"idx_to_emotion_class: {fer.idx_to_emotion_class}")
    print(f"情绪全局均值: {global_emo_mean}")
    print(f"情绪全局预测: {fer.idx_to_emotion_class[global_emo_idx]}")
    print("="*50 + "\n")

    # 构建逐帧映射
    frame_emotion_map = {}
    frame_status_map  = {}

    for i, data in enumerate(face_data_list):
        frame_num = data["frame_num"]

        # 情绪：当前帧最高分
        top_emo_idx = np.argmax(emotion_scores[i])
        emotion_name = fer.idx_to_emotion_class.get(top_emo_idx, "Unknown")
        frame_emotion_map[frame_num] = emotion_name

        # 专注度：对齐到正确的窗口索引
        # engagement_labels[k] 对应 face_data_list[k + window_size - 1]
        eng_offset = i - (window_size - 1)
        if 0 <= eng_offset < eng_count:
            label    = engagement_labels[eng_offset]
            cn_label = "专注" if label == "Engaged" else "不专注"
            frame_status_map[frame_num] = {"box": data["box"], "label": cn_label}
        # 前 window_size-1 帧没有专注度预测（窗口未填满），不写入 frame_status_map

    del all_faces, face_data_list
    print(f"✅ 专注度预测完成，滑动窗口大小: {window_size} 帧")
    
    return frame_status_map, frame_emotion_map, (w, h, fps, total_frames)

# ===== 配置参数 =====
TEST_VIDEO = "/kaggle/input/datasets/liuweiq/daxiaonailong/me-k80.mp4"
WINDOW_SIZE = 128  # 官方推荐值，必须足够大才能捕捉时序专注信息

print("=" * 50)
print("开始提取人脸并预测专注度...")
print("=" * 50)

# 运行：提取人脸并预测专注度
result = extract_faces_and_predict(TEST_VIDEO, window_size=WINDOW_SIZE)

if result is not None:
    frame_status_map, frame_emotion_map, video_info = result
    print(f"\n✅ 人脸提取完成！共检测 {len(frame_status_map)} 帧")
else:
    print("❌ 处理失败：未能检测到人脸")

# %% [code]
# 7. 定义并运行：渲染视频并标注
def render_video_with_annotations(video_path, frame_status_map, frame_emotion_map, video_info, output_path="output.mp4"):
    """第二趟：逐帧绘制人脸框、专注度标签和情绪标签"""
    w, h, fps, total_frames = video_info
    
    # 情绪中英对照
    emotion_cn = {
        'Neutral': '中性', 'Happiness': '高兴', 'Surprise': '惊讶',
        'Sadness': '悲伤', 'Anger': '生气', 'Disgust': '厌恶',
        'Fear': '恐惧', 'Contempt': '轻蔑'
    }
    
    print("正在初始化视频编码器...")
    
    # 测试可用编码器
    codec_options = [
        ('XVID', cv2.VideoWriter_fourcc(*'XVID'), '.avi'),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG'), '.avi'),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'),
    ]
    
    writer = None
    for codec_name, fourcc, ext in codec_options:
        test_file = f"test_{codec_name.lower()}{ext}"
        test_writer = cv2.VideoWriter(test_file, fourcc, 30, (320, 240))
        
        try:
            test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            test_writer.write(test_frame)
            test_writer.write(test_frame)
            test_writer.write(test_frame)
            time.sleep(0.5)
            test_writer.release()
            
            if os.path.exists(test_file) and os.path.getsize(test_file) > 100:
                os.remove(test_file)
                
                if ext == '.avi':
                    output_path = output_path.replace('.mp4', f'_{codec_name.lower()}.avi')
                
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if writer.isOpened():
                    print(f"✅ 使用编码器 {codec_name}，输出: {output_path}")
                    break
        except Exception as e:
            print(f"⚠️ 编码器 {codec_name} 失败: {e}")
            test_writer.release() if 'test_writer' in dir() else None
            if os.path.exists(test_file):
                os.remove(test_file)
    
    if writer is None:
        print("❌ 所有编码器都失败")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法重新打开视频")
        writer.release()
        return None

    frame_idx = 0
    last_face = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        draw_frame = frame.copy()
        status = None
        
        # 查找当前帧状态
        if frame_idx in frame_status_map:
            status = frame_status_map[frame_idx]
            last_face = {'box': status['box'], 'label': status['label']}
        elif last_face is not None:
            status = last_face

        # 绘制标注
        if status is not None:
            x1, y1, x2, y2 = status["box"]
            label = status["label"]
            color_bgr = (0, 255, 0) if label == "专注" else (0, 0, 255)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color_bgr, 3)
            
            # 使用PIL绘制中文
            frame_rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            draw_pil = ImageDraw.Draw(img_pil)
            
            # 获取当前帧的情绪
            emotion_name = frame_emotion_map.get(frame_idx, None)
            if emotion_name:
                emotion_text = f"情绪: {emotion_cn.get(emotion_name, emotion_name)}"
            else:
                emotion_text = "情绪: 未知"
            
            # 绘制专注度标签
            text = f"状态：{label}"
            bbox = draw_pil.textbbox((0, 0), text, font=chinese_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # 第一行：专注度状态（蓝色背景）
            draw_pil.rectangle([x1, y1 - text_h - 15, x1 + text_w + 10, y1], fill=color_rgb)
            draw_pil.text((x1 + 5, y1 - text_h - 12), text, font=chinese_font, fill=(0, 0, 0))
            
            # 第二行：情绪标签（紫色背景）
            bbox2 = draw_pil.textbbox((0, 0), emotion_text, font=chinese_font)
            text2_w = bbox2[2] - bbox2[0]
            text2_h = bbox2[3] - bbox2[1]
            
            emotion_color = (200, 150, 255)  # 紫色
            draw_pil.rectangle([x1, y1 - text_h - 15 - text2_h - 12, x1 + text2_w + 10, y1 - text_h - 15], fill=emotion_color)
            draw_pil.text((x1 + 5, y1 - text_h - 15 - text2_h - 10), emotion_text, font=chinese_font, fill=(0, 0, 0))
            
            frame_rgb = np.array(img_pil)
            draw_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        writer.write(draw_frame)

        if frame_idx % 200 == 0:
            print(f"渲染进度: {frame_idx}/{total_frames} 帧", end='\r')

    cap.release()
    writer.release()
    print(f"\n✅ 视频渲染完成: {output_path}")
    return output_path

# 运行：渲染视频
OUTPUT_VIDEO = "engagement_result.mp4"

if result is not None:
    print("=" * 50)
    print("开始渲染视频...")
    print("=" * 50)
    
    output = render_video_with_annotations(
        TEST_VIDEO, 
        frame_status_map,
        frame_emotion_map,
        video_info, 
        output_path=OUTPUT_VIDEO
    )
    
    if output:
        print(f"\n✅ 渲染完成！临时文件: {output}")
else:
    print("❌ 跳过渲染：未检测到人脸")

# %% [code]
# 8. 定义并运行：FFmpeg重新编码（压缩+保留音频）
def ffmpeg_reencode(video_path, original_video_path, crf=28):
    """使用FFmpeg重新编码视频"""
    try:
        subprocess.run('ffmpeg -version', shell=True, capture_output=True, check=True)
    except:
        print("⚠️ FFmpeg不可用，跳过重新编码")
        return video_path

    print("✅ FFmpeg可用，开始重新编码...")
    
    final_output = video_path.replace('.avi', '_ffmpeg.mp4')
    temp_output = video_path.replace('.avi', '_temp.avi')
    
    shutil.move(video_path, temp_output)
    
    # 尝试保留音频
    cmd = f'ffmpeg -i {temp_output} -i "{original_video_path}" -map 0:v:0 -map 1:a:0? -c:v libx264 -crf {crf} -c:a aac -shortest "{final_output}" -y'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✅ FFmpeg编码成功（已保留音频）: {final_output}")
            os.remove(temp_output)
            return final_output
    except:
        pass
    
    # 降级：无音频编码
    cmd2 = f'ffmpeg -i {temp_output} -c:v libx264 -crf {crf} -an "{final_output}" -y'
    try:
        result = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✅ FFmpeg编码成功（无音频）: {final_output}")
            os.remove(temp_output)
            return final_output
    except:
        pass
    
    print("⚠️ FFmpeg编码失败，使用原始输出")
    shutil.move(temp_output, video_path)
    return video_path

# 运行：FFmpeg编码
CRF = 28  # FFmpeg压缩质量 (0-51, 越小越好)

if 'output' in dir() and output:
    print("=" * 50)
    print("开始FFmpeg压缩编码...")
    print("=" * 50)
    
    final = ffmpeg_reencode(output, TEST_VIDEO, crf=CRF)
    
    print("=" * 50)
    print(f"✅ 处理完成！最终输出: {final}")
    print("=" * 50)
else:
    print("❌ 跳过FFmpeg编码：无渲染输出")

# %% [code]
# 9. 展示最终视频
from IPython.display import Video, display

# 最终输出文件路径
display_video_path = final if 'final' in dir() and final else output
display_video_path = display_video_path.replace('.avi', '.mp4')

print(f"展示视频: {display_video_path}")
display(Video(display_video_path, embed=True, width=800))
