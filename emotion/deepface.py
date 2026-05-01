# %% [code] {"execution":{"iopub.status.busy":"2026-04-29T15:12:48.143935Z","iopub.execute_input":"2026-04-29T15:12:48.144285Z","iopub.status.idle":"2026-04-29T15:12:56.971173Z","shell.execute_reply.started":"2026-04-29T15:12:48.144253Z","shell.execute_reply":"2026-04-29T15:12:56.970258Z"},"jupyter":{"outputs_hidden":false}}
from IPython.display import clear_output  # 这行必须加！
!pip install deepface ultralytics
clear_output()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-29T15:12:56.973138Z","iopub.execute_input":"2026-04-29T15:12:56.973527Z","iopub.status.idle":"2026-04-29T15:13:09.756622Z","shell.execute_reply.started":"2026-04-29T15:12:56.973499Z","shell.execute_reply":"2026-04-29T15:13:09.755850Z"}}
!apt-get update && apt-get install -y fonts-wqy-zenhei

# %% [code] {"execution":{"iopub.status.busy":"2026-04-29T15:13:09.757728Z","iopub.execute_input":"2026-04-29T15:13:09.758398Z","iopub.status.idle":"2026-04-29T15:13:32.790151Z","shell.execute_reply.started":"2026-04-29T15:13:09.758354Z","shell.execute_reply":"2026-04-29T15:13:32.789245Z"},"jupyter":{"outputs_hidden":false}}
from deepface import DeepFace

# %% [code] {"execution":{"iopub.status.busy":"2026-04-29T15:13:32.791421Z","iopub.execute_input":"2026-04-29T15:13:32.792104Z","iopub.status.idle":"2026-04-29T15:15:04.740447Z","shell.execute_reply.started":"2026-04-29T15:13:32.792080Z","shell.execute_reply":"2026-04-29T15:15:04.739572Z"},"jupyter":{"outputs_hidden":false}}
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
from moviepy.editor import VideoFileClip
import os
import math
from collections import deque
from statistics import mode, mean

# --- 1. 定义情绪的中英文翻译字典 ---
EMOTION_DICT = {
    'angry': '愤怒',
    'disgust': '厌恶',
    'fear': '恐惧',
    'happy': '开心',
    'sad': '悲伤',
    'surprise': '惊讶',
    'neutral': '平静'
}

# --- 1.1 定义情绪对应的颜色 (BGR格式) ---
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # 红色 - 愤怒
    'disgust': (0, 140, 255),   # 橙色 - 厌恶
    'fear': (0, 165, 255),      # 深橙色 - 恐惧
    'happy': (0, 255, 0),       # 绿色 - 开心
    'sad': (255, 0, 0),         # 蓝色 - 悲伤
    'surprise': (255, 255, 0),   # 青色 - 惊讶
    'neutral': (255, 255, 255)   # 白色 - 平静
}

# --- 1.2 定义情绪对应的颜文字 ---
# 已移除颜文字

# --- 2. 封装 OpenCV 绘制中文的函数 ---
def cv2_add_chinese_text_with_bg(img, text, position, text_color=(0, 255, 0), text_size=20, bg_alpha=0.5):
    """
    绘制带浅透明背景的中文文字
    :param img: OpenCV图像
    :param text: 要绘制的文字
    :param position: 文字位置 (x, y)
    :param text_color: 文字颜色 (B, G, R)
    :param text_size: 文字大小
    :param bg_alpha: 背景透明度 (0-1, 0为完全透明, 1为不透明)
    :return: 绘制后的图像
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    
    try:
        font = ImageFont.truetype(font_path, text_size, encoding="utf-8")
    except IOError:
        print(f"找不到字体文件 {font_path}，请先上传字体文件！")
        return img 
    
    # 获取文字边界框
    bbox = draw.textbbox(position, text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 添加一些padding
    padding = 8
    bg_x1 = bbox[0] - padding
    bg_y1 = bbox[1] - padding
    bg_x2 = bbox[2] + padding
    bg_y2 = bbox[3] + padding
    
    # 圆角半径
    corner_radius = 8
    
    # 创建透明背景层
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 绘制半透明背景 (黑色圆角矩形背景)
    bg_color = (0, 0, 0, int(255 * bg_alpha))
    overlay_draw.rounded_rectangle([bg_x1, bg_y1, bg_x2, bg_y2], radius=corner_radius, fill=bg_color)
    
    # 添加细边框（颜色与文字一致，增加高级感）
    outline_color = (text_color[2], text_color[1], text_color[0], int(255 * 0.8))  # 转换为RGBA
    overlay_draw.rounded_rectangle([bg_x1, bg_y1, bg_x2, bg_y2], radius=corner_radius, outline=outline_color, width=1)
    
    # 合并背景层到原图
    img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay)
    img_pil = img_pil.convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    
    # 绘制文字
    b, g, r = text_color
    draw.text(position, text, (r, g, b), font=font)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=20):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    
    try:
        font = ImageFont.truetype(font_path, text_size, encoding="utf-8")
    except IOError:
        print(f"找不到字体文件 {font_path}，请先上传字体文件！")
        return img 
        
    b, g, r = text_color
    draw.text(position, text, (r, g, b), font=font)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def process_video_every_frame(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_output = "temp_no_audio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_count = 0
    
    # --- 新增：用于平滑数据的历史轨迹配置 ---
    face_tracks = []
    SMOOTH_FRAMES = 3  # 缓存的帧数（15帧约0.5秒），数值越大变化越迟钝但越稳定
    MAX_DIST = 100      # 判断是否为同一个人的最大允许位移（像素）

    print("开始逐帧分析视频（警告：这将会非常耗时）...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        try:
            results = DeepFace.analyze(
                img_path=frame, 
                actions=['age', 'emotion'], 
                enforce_detection=False,
                detector_backend='yolov8n',
                align=True
            )
            
            current_tracks = []
            
            for face_data in results:
                region = face_data['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # 计算人脸中心点
                cx, cy = x + w / 2, y + h / 2
                raw_emotion = face_data['dominant_emotion']
                age = face_data['age']
                
# --- 新增：获取当前情绪的置信度分数 ---
                # DeepFace 的 emotion 键包含所有情绪的百分比分数
                emotion_score = face_data['emotion'][raw_emotion] 
                
                # 寻找最近的已知人脸轨迹
                best_match = None
                min_dist = float('inf')
                for track in face_tracks:
                    dist = math.hypot(cx - track['center'][0], cy - track['center'][1])
                    if dist < min_dist and dist < MAX_DIST:
                        min_dist = dist
                        best_match = track
                        
                # 更新或创建轨迹，并计算平滑结果
                if best_match:
                    best_match['center'] = (cx, cy)
                    # 将情绪及其置信度打包存入队列
                    best_match['emotions'].append((raw_emotion, emotion_score))
                    best_match['ages'].append(age)
                    
                    # --- 核心优化：根据置信度累加判断哪个情绪更好 ---
                    emotion_weights = {}
                    for em, score in best_match['emotions']:
                        emotion_weights[em] = emotion_weights.get(em, 0) + score
                    # 取累积置信度最高的情绪
                    final_emotion = max(emotion_weights, key=emotion_weights.get)
                    
                    final_age = int(mean(best_match['ages']))
                    current_tracks.append(best_match)
                else:
                    # 如果是新出现的人脸，初始化队列
                    new_track = {
                        'center': (cx, cy),
                        'emotions': deque([(raw_emotion, emotion_score)], maxlen=SMOOTH_FRAMES),
                        'ages': deque([age], maxlen=SMOOTH_FRAMES)
                    }
                    current_tracks.append(new_track)
                    final_emotion = raw_emotion
                    final_age = int(age)
                
                # 画人脸框（根据情绪使用不同颜色）
                color = EMOTION_COLORS.get(final_emotion, (0, 255, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # 提取并翻译最终的平滑情绪
                cn_emotion = EMOTION_DICT.get(final_emotion, '未知')
                
                # 获取颜文字（已禁用）
                # emoticon = EMOTION_EMOTICONS.get(final_emotion, '')
                
                info_text = [
                    f"情绪: {cn_emotion}",
                    f"年龄: {final_age}"
                ]
                # --- 优化后的文字防越界绘制逻辑 ---
                line_height = 25  # 每行文字的高度
                total_text_height = len(info_text) * line_height
                
                # 1. 动态决定 Y 坐标（上下防越界）
                if y >= total_text_height:
                    # 如果上方空间足够，写在人脸框上方
                    start_y = y - total_text_height
                elif y + h + total_text_height < frame.shape[0]:
                    # 如果上方不够，但下方空间足够，写在人脸框下方
                    start_y = y + h + 5
                else:
                    # 如果上下都不够（极端情况），强制写在画面顶部边缘
                    start_y = 5

                # 2. 动态决定 X 坐标（左右防越界）
                # 防止 x 为负数导致文字在左侧消失
                safe_x = max(0, x)
                
                # 计算所有文字的整体背景框
                font_size = 20
                font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
                try:
                    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                except IOError:
                    font = ImageFont.load_default()
                
                # 测量每行文字宽度
                img_temp = Image.new('RGB', (1, 1))
                draw_temp = ImageDraw.Draw(img_temp)
                
                max_text_width = 0
                for text in info_text:
                    bbox = draw_temp.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    max_text_width = max(max_text_width, text_width)
                
                # 计算背景框的位置和大小
                padding = 8
                bg_x1 = safe_x - padding
                bg_y1 = int(start_y) - padding
                bg_x2 = safe_x + max_text_width + padding
                bg_y2 = int(start_y + total_text_height) + padding
                
                # 在图像上绘制圆角背景框
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
                overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # 绘制半透明圆角矩形背景
                bg_color = (0, 0, 0, int(255 * 0.6))
                overlay_draw.rounded_rectangle([bg_x1, bg_y1, bg_x2, bg_y2], radius=8, fill=bg_color)
                
                # 添加细边框
                outline_color = (255, 255, 255, int(255 * 0.8))
                overlay_draw.rounded_rectangle([bg_x1, bg_y1, bg_x2, bg_y2], radius=8, outline=outline_color, width=1)
                
                # 合并背景
                img_pil = Image.alpha_composite(img_pil, overlay)
                frame = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                
                # 绘制所有文字
                for i, text in enumerate(info_text):
                    pos_y = int(start_y + (i * line_height))
                    frame = cv2_add_chinese_text(frame, text, (safe_x, pos_y), text_color=(255, 255, 255), text_size=20)
            
            # 更新全局轨迹（丢弃当前帧未检测到的人脸）
            face_tracks = current_tracks
                            
        except Exception as e:
            print(f"第 {frame_count} 帧报错: {e}")

        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"已深度分析 {frame_count} 帧...")

    cap.release()
    out.release()

    print("画面处理完毕，正在合成音频...")
    try:
        video_clip = VideoFileClip(temp_output)
        original_video = VideoFileClip(input_path)
        
        final_video = video_clip.set_audio(original_video.audio)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        video_clip.close()
        original_video.close()
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"处理完成！最终视频保存至: {output_path}")
        
    except Exception as e:
        print(f"音频合成失败: {e}")

# 运行测试
input_file = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuqiangdong.mp4"
output_file = "result_every_frame.mp4"
process_video_every_frame(input_file, output_file)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# 展示处理后的视频
from IPython.display import Video
import os

# 压缩视频以便展示（如果原视频较大的话）
compressed_output_path = 'compressed_result.mp4'
if os.path.exists(output_file):
    print("正在压缩视频...")
    os.system(f'ffmpeg -i {output_file} -vcodec libx264 -crf 28 {compressed_output_path} -y')
    
    # 展示压缩后的视频
    print("展示视频:")
    display(Video(compressed_output_path, embed=True))
else:
    print(f"输出文件 {output_file} 不存在，无法展示视频")
