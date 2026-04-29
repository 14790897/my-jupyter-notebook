# %% [code] {"execution":{"iopub.status.busy":"2026-04-27T15:39:33.064339Z","iopub.execute_input":"2026-04-27T15:39:33.065251Z","iopub.status.idle":"2026-04-27T15:39:37.959736Z","shell.execute_reply.started":"2026-04-27T15:39:33.065218Z","shell.execute_reply":"2026-04-27T15:39:37.958932Z"}}
from IPython.display import clear_output  # 这行必须加！
!pip install emotiefflib[all]  ultralytics moviepy

clear_output()

# %% [code] {"execution":{"iopub.status.busy":"2026-04-27T15:39:37.961305Z","iopub.execute_input":"2026-04-27T15:39:37.961614Z","iopub.status.idle":"2026-04-27T15:39:37.971053Z","shell.execute_reply.started":"2026-04-27T15:39:37.961583Z","shell.execute_reply":"2026-04-27T15:39:37.970504Z"}}
import os
import urllib.request

def ensure_face_model(model_path="yolov8n-face.pt"):
    """检查并自动下载 YOLOv8 人脸模型"""
    if os.path.exists(model_path):
        return True
        
    print(f"未找到 {model_path}，正在自动下载 (约 6MB)...")
    # 使用国内访问极其稳定的 HuggingFace 镜像源 (常用于 ADetailer 插件)
    url = "https://huggingface.co/deepghs/yolo-face/resolve/739664f2d00e436a8882238f83175ab0f6497578/yolov8n-face/model.pt"
    
    try:
        # 添加简单的请求头防止被拦截
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("✅ 模型下载成功！\n")
        return True
    except Exception as e:
        print(f"❌ 自动下载失败: {e}")
        print(f"请手动点击链接下载，并把文件放到代码所在文件夹: {url}")
        return False

# 在调用 analyze_video_engagement 之前，先运行检查
ensure_face_model("yolov8n-face.pt")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-28T12:28:09.342680Z","iopub.execute_input":"2026-04-28T12:28:09.343674Z","iopub.status.idle":"2026-04-28T12:29:10.585310Z","shell.execute_reply.started":"2026-04-28T12:28:09.343636Z","shell.execute_reply":"2026-04-28T12:29:10.584641Z"}}
import cv2
import numpy as np
from ultralytics import YOLO
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
# remove MoviePy dependencies
# from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont

def process_and_render_video(
    video_path, 
    output_path="engagement_result.mp4", 
    face_model_path="yolov8n-face.pt", 
    window_size=12,
    font_path="simhei.ttf"  # 请上传字体文件（例如 simhei.ttf）并在此指定路径
):
    """
    第一趟：全量提取人脸并批量预测专注度
    计算：特征提取、分类、滑动窗口平滑得分
    第二趟：重新读取视频，精准逐帧查找结果和框，绘制（含中文）并导出 (使用 OpenCV 逐帧，解决延迟和消失)
    """
    # ==========================================
    # 阶段 1：初始化与模型加载
    # ==========================================
    print("正在加载 YOLOv8 人脸检测模型...")
    try:
        # 使用 yolov8n-face.pt 精度更高，更专注于人脸
        face_detector = YOLO(face_model_path)
    except Exception as e:
        print(f"加载 YOLOv8 模型失败: {e}")
        return

    print("正在加载 EmotiEffLib 模型...")
    fer = EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_best_vgaf", device="cuda")

    # 尝试加载中文字体
    try:
        # 字体大小设为 30，可根据视频分辨率自行调整
        chinese_font = ImageFont.truetype(font_path, 30) 
    except IOError:
        print(f"⚠️ 警告: 找不到字体文件 '{font_path}'。将使用默认字体（可能导致中文显示为方块）。")
        print("💡 提示: 如果在本地 Windows 上运行，可以改成 'C:/Windows/Fonts/msyh.ttc' (微软雅黑)。")
        print("💡 提示: 如果在 Kaggle 上运行，建议上传一个 simhei.ttf 并在 font_path 中指定路径。")
        chinese_font = ImageFont.load_default()

    # ==========================================
    # 阶段 2：第一趟读取（提取人脸）
    # ==========================================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取输入视频属性，用于第二趟写出时对齐
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"第一趟扫描：开始提取人脸。总帧数: {total_frames}, FPS: {fps:.2f}")

    face_data_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = face_detector(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) > 0:
            # 取面积最大的脸（如果有多个人脸）
            if len(boxes) > 1:
                boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            
            x1, y1, x2, y2 = map(int, boxes[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_bgr = frame[y1:y2, x1:x2]
            if face_bgr.shape[0] >= 20 and face_bgr.shape[1] >= 20:
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                
                # 存储该帧信息，帧号从 1 开始
                face_data_list.append({
                    "frame_num": frame_count,
                    "box": (x1, y1, x2, y2),
                    "face_rgb": face_rgb
                })

        if frame_count % 200 == 0:
            print(f"提取进度: {frame_count}/{total_frames} 帧")

    cap.release()
    print(f"第一趟完成，共提取到 {len(face_data_list)} 帧人脸。")

    # ==========================================
    # 阶段 3：特征提取与批量预测专注度（保留原逻辑）
    # ==========================================
    if len(face_data_list) == 0:
        print("未检测到人脸，无法分析。")
        return

    print("正在提取人脸特征并分类...")
    all_faces = [data["face_rgb"] for data in face_data_list]
    
    features = fer.extract_features(all_faces)
    _, emotion_scores = fer.classify_emotions(features, logits=True)
    _, engagement_scores = fer.classify_engagement(features)
    
    print(f"特征处理完成，开始按滑动窗口({window_size}帧)构建帧状态映射表...")

    # 用于第二趟逐帧查找和绘制的映射表 (帧号 -> {框, 标签})
    frame_status_map = {}
    
    for i, data in enumerate(face_data_list):
        frame_num = data["frame_num"]
        
        # 确定当前滑动窗口的范围
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        
        # 计算滑动窗口内得分平均值
        window_engagement_scores = engagement_scores[start_idx:end_idx]
        engagement_score_mean = np.mean(window_engagement_scores, axis=0)
        
        # 取得分最高的索引
        engagement_idx = np.argmax(engagement_score_mean)
        engagement_label_eng = fer.idx_to_engagement_class[engagement_idx]

        # 转换为中文标签
        if "Not" in engagement_label_eng or "Distracted" in engagement_label_eng or "not" in engagement_label_eng:
            cn_label = "不专注"
        else:
            cn_label = "专注"
        
        frame_status_map[frame_num] = {
            "box": data["box"],
            "label": cn_label
        }

    del all_faces
    del face_data_list
    # del features # 视内存情况可手动释放

    # ==========================================
    # 阶段 4：OpenCV 逐帧处理与写出视频 (解决同步和消失)
    # ==========================================
    print(f"\n第二趟扫描：启动 OpenCV 引擎精准渲染视频...")
    
    # 初始化输出视频写出流 ('mp4v' 或 'avc1')
    # 请确保 output_path 的扩展名和 codec 匹配
    # (cv2.VideoWriter_fourcc(*'mp4v') 一般生成 .mp4)
    # (cv2.VideoWriter_fourcc(*'avc1') 兼容性更好，常用于 .mp4/.m4v)
    output_extension = output_path.split('.')[-1]
    if output_extension.lower() == 'mp4':
        output_codec = cv2.VideoWriter_fourcc(*'avc1')
    else:
        # fallback codec
        output_codec = cv2.VideoWriter_fourcc(*'mp4v') 

    writer = cv2.VideoWriter(output_path, output_codec, fps, (w, h))
    if not writer.isOpened():
        print(f"无法创建输出视频文件: {output_path}")
        return

    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        print(f"第二趟无法重新打开视频: {video_path}")
        writer.release()
        return

    current_frame_idx_draw = 0
    
    # 用于记录前一帧有人脸的框信息，以便在未检测到时进行预测/平滑
    # format: { 'box': (x1, y1, x2, y2), 'label': '专注'/'不专注' }
    last_frame_face = None

    while cap2.isOpened():
        ret, frame_cv_bgr = cap2.read()
        if not ret:
            break
        current_frame_idx_draw += 1 # 帧号递增

        draw_img_bgr = frame_cv_bgr.copy() # 拷贝帧用于绘制，保留原图
        
        status = None
        
        # 1. 尝试从 frame_status_map 精准查找当前帧结果
        if current_frame_idx_draw in frame_status_map:
            status = frame_status_map[current_frame_idx_draw]
            # 更新有人脸的帧框和标签信息
            last_frame_face = { 'box': status['box'], 'label': status['label'] }
        
        # 2. 如果当前帧未检测到人脸，尝试使用前一帧框位置预测/平滑位置
        elif last_frame_face is not None:
            # 这是一个非常简单的“前向平滑预测”，只用前一帧的框位置和标签
            # 在视频尾部人脸还在但由于各种原因漏检时，能有效防止框和结果消失。
            status = last_frame_face 
        
        # 3. 如果能查找到状态或有预测结果，则绘制
        if status is not None:
            x1, y1, x2, y2 = status["box"]
            label = status["label"]

            # 颜色设置：绿色=专注，红色=不专注 (BGR 格式，用于 OpenCV 绘图)
            color_bgr = (0, 255, 0) if label == "专注" else (0, 0, 255)

            # --- PIL 绘制中文 ---
            # PIL 颜色需用 RGB
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) 

            # 绘制 OpenCV 人脸框
            cv2.rectangle(draw_img_bgr, (x1, y1), (x2, y2), color_bgr, 3)
            
            # 转换到 PIL Image 以绘制中文
            # OpenCV 是 BGR，需先转 RGB
            frame_cv_rgb = cv2.cvtColor(draw_img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_cv_rgb)
            draw_pil = ImageDraw.Draw(img_pil)
            
            text = f"状态：{label}"
            
            # 获取文字的宽高，用于绘制背景色块
            bbox = draw_pil.textbbox((0, 0), text, font=chinese_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # 在文字下方画一个实心矩形背景，确保文字清晰可见
            # PIL rectangle 坐标格式为 [x0, y0, x1, y1]
            # 背景色块颜色和框颜色一致
            draw_pil.rectangle([x1, y1 - text_h - 15, x1 + text_w + 10, y1], fill=color_rgb)
            
            # 绘制纯黑色的中文文字
            draw_pil.text((x1 + 5, y1 - text_h - 12), text, font=chinese_font, fill=(0, 0, 0))
            
            # 将画好中文的 PIL Image 重新转回 OpenCV BGR 数组
            frame_pil_rgb = np.array(img_pil)
            draw_img_bgr = cv2.cvtColor(frame_pil_rgb, cv2.COLOR_RGB2BGR)

        # 4. 将绘制好或未绘制人脸框的原始帧写出视频
        writer.write(draw_img_bgr)

        # 简单的进度打印
        if current_frame_idx_draw % 200 == 0:
            print(f"渲染进度: {current_frame_idx_draw}/{total_frames} 帧", end='\r')

    # 释放资源
    cap2.release()
    writer.release()

    # OpenCV 不会自动复制音轨，如果需要，可以使用 moviepy 复制，这里为了简洁只保留最关键的修复
    print(f"\n✅ 导出完成！视频文件已保存至: {output_path}")

# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    # 请根据实际环境修改测试视频和输出视频路径
    TEST_VIDEO = "/kaggle/input/datasets/liuweiq/daxiaonailong/WIN_20260428_23_02_08_Pro.mp4" 
    OUTPUT_VIDEO = "engagement_final.mp4"
    
    # ⚠️ 重要提示：请将下面的 FONT_FILE 修改为你实际环境中的中文字体文件路径
    # 如果你在本地 Windows 上运行，可以改成 "C:/Windows/Fonts/msyh.ttc" (微软雅黑)
    # 如果在 Kaggle 运行，建议上传一个 simhei.ttf 并在 font_path 中指定路径。
    # 这里设置默认路径，如果上传了 simhei.ttf 放在同一目录下即可跑通。
    FONT_FILE = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc" 

    process_and_render_video(
        TEST_VIDEO, 
        output_path=OUTPUT_VIDEO, 
        window_size=12,
        font_path=FONT_FILE
    )
