# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-23T03:53:38.961787Z","iopub.execute_input":"2026-04-23T03:53:38.962130Z","iopub.status.idle":"2026-04-23T03:53:55.664549Z","shell.execute_reply.started":"2026-04-23T03:53:38.962086Z","shell.execute_reply":"2026-04-23T03:53:55.663686Z"}}
from IPython.display import clear_output
!pip install gliner2 -U
!pip install srt
!pip install opencv-python moviepy pillow -U
clear_output()

# %% [code] {"execution":{"iopub.status.busy":"2026-04-23T03:53:55.666668Z","iopub.execute_input":"2026-04-23T03:53:55.667076Z","iopub.status.idle":"2026-04-23T03:54:40.547815Z","shell.execute_reply.started":"2026-04-23T03:53:55.667044Z","shell.execute_reply":"2026-04-23T03:54:40.547167Z"},"jupyter":{"outputs_hidden":false}}
from pathlib import Path
import srt
import warnings
import cv2
import numpy as np
warnings.filterwarnings("ignore")

from gliner2 import GLiNER2
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip

# ===================== 【修复1】只加载一次模型 =====================
extractor = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-23T03:54:40.548746Z","iopub.execute_input":"2026-04-23T03:54:40.549410Z","iopub.status.idle":"2026-04-23T03:54:40.839038Z","shell.execute_reply.started":"2026-04-23T03:54:40.549382Z","shell.execute_reply":"2026-04-23T03:54:40.838172Z"},"jupyter":{"outputs_hidden":false}}
from transformers import pipeline
# sentiment_model  = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
# emotion_model = pipeline(
#     "text-classification",
#     model="bardsai/chinese-emotion-classification",
#     device=0
# )

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-23T03:54:40.840171Z","iopub.execute_input":"2026-04-23T03:54:40.840601Z","iopub.status.idle":"2026-04-23T03:54:40.856492Z","shell.execute_reply.started":"2026-04-23T03:54:40.840562Z","shell.execute_reply":"2026-04-23T03:54:40.855472Z"}}
# ===================== 【修复2】定义统一 Schema =====================
# 实体 + 情感 一次提取（官方标准用法，识别率最高）
schema = (
    extractor.create_schema()
    .entities([
     "货币金额"
    ])
        .classification("情感", [
        "高兴",
        "生气",
        "不满",
        "疑问",
        "平静",
        "嘲讽",
        "惊讶"
    ])
)

def format_timedelta_srt(td):
    total_ms = int(td.total_seconds() * 1000)
    hours, remain = divmod(total_ms, 3600 * 1000)
    minutes, remain = divmod(remain, 60 * 1000)
    seconds, milliseconds = divmod(remain, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def timedelta_to_seconds(td):
    return td.total_seconds()

def recognize_srt_sentences(srt_path: str):
    srt_file = Path(srt_path)
    raw_content = srt_file.read_text(encoding="utf-8")
    subtitles = list(srt.parse(raw_content))

    all_results = []
    
    # ===================== 【修复3】降低阈值，提高召回 =====================
    for sub in subtitles:
        sentence = sub.content.strip().replace("\n", " ")
        if not sentence:
            continue

        # 【官方正确用法】一次提取 实体 + 情感
        result = extractor.extract(
            sentence, 
            schema,
            threshold=0.35,    # 降低！更容易识别
        )

        entities = result.get("entities", {})
        currency_entities = None
        if isinstance(entities, dict):
            currency_amount = entities.get("货币金额")
            if currency_amount:
                currency_entities = {"货币金额": currency_amount}
        elif isinstance(entities, list):
            filtered_currency_entities = [
                x
                for x in entities
                if isinstance(x, dict)
                and (x.get("label") == "货币金额" or x.get("entity") == "货币金额")
            ]
            if filtered_currency_entities:
                currency_entities = filtered_currency_entities

        has_entities = currency_entities is not None
        sentiment_value = result.get("情感")
        has_sentiment = bool(sentiment_value)

        # ==================== 新加：调用精准情感模型 ====================
        # real_emotion = sentiment_model(sentence)[0]

        item = {
            "index": sub.index,
            "start": format_timedelta_srt(sub.start),
            "end": format_timedelta_srt(sub.end),
            "start_sec": timedelta_to_seconds(sub.start),
            "end_sec": timedelta_to_seconds(sub.end),
            "text": sentence,
            # "real_emotion": real_emotion  # 把精准情感也存起来
        }
        if has_entities:
            item["entities"] = currency_entities
        if has_sentiment:
            item["sentiment"] = sentiment_value
        all_results.append(item)

        # 输出（完全保留你原来的格式，只多加一行）
        print(f"【{sub.index}】{item['start']} --> {item['end']}")
        print(f"内容：{sentence}")
        if "entities" in item:
            print(f"实体：{item['entities']}")
        if "sentiment" in item:
            print(f"GLiNER情感：{item['sentiment']}")       # 原来的
        # print(f"✅ 真实情感：{item['real_emotion']}")    # 新加的（准）
        print("-" * 60)

    return all_results


def get_active_result(results, t_sec):
    for item in results:
        if item["start_sec"] <= t_sec <= item["end_sec"]:
            return item
    return None


def build_overlay_text(item):
    lines = [
        f"字幕: {item['text']}",
    ]
    if item.get("entities"):
        lines.append(f"实体: {item['entities']}")
    if item.get("sentiment"):
        lines.append(f"GLiNER情感: {item['sentiment']}")
    # lines.append(f"真实情感: {item.get('real_emotion', {})}")
    return "\n".join(lines)


def draw_text_top_left(frame_bgr, text, font_path=None, font_size=20):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    x, y = 20, 20
    lines = text.split("\n")
    line_h = font_size + 10
    # box_w = image.width - 40
    # box_h = 20 + line_h * len(lines)

    # draw.rounded_rectangle(
    #     [(x - 10, y - 10), (x - 10 + box_w, y - 10 + box_h)],
    #     radius=12,
    #     fill=(0, 0, 0, 140),
    # )

    vivid_color = (255, 235, 59, 255)

    for i, line in enumerate(lines):
        draw.text(
            (x, y + i * line_h),
            line,
            font=font,
            fill=vivid_color,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    out_rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def render_video_with_results(video_path, results, output_video_path, font_path=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video_path = str(Path(output_video_path).with_name("temp_no_audio.mp4"))
    writer = cv2.VideoWriter(
        temp_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = frame_idx / fps if fps > 0 else 0.0
        active = get_active_result(results, t_sec)
        if active is not None:
            overlay_text = build_overlay_text(active)
            frame = draw_text_top_left(frame, overlay_text, font_path=font_path, font_size=20)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    original_clip = VideoFileClip(video_path)
    silent_clip = VideoFileClip(temp_video_path)
    final_clip = silent_clip.with_audio(original_clip.audio)
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

    original_clip.close()
    silent_clip.close()
    final_clip.close()

# %% [code] {"execution":{"iopub.status.busy":"2026-04-23T03:54:40.857516Z","iopub.execute_input":"2026-04-23T03:54:40.858326Z","iopub.status.idle":"2026-04-23T03:54:45.753155Z","shell.execute_reply.started":"2026-04-23T03:54:40.858298Z","shell.execute_reply":"2026-04-23T03:54:45.752159Z"},"jupyter":{"outputs_hidden":false}}
from transformers import pipeline

sentiment = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
print(sentiment("这个AI产品体验非常差"))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-23T03:54:45.754275Z","iopub.execute_input":"2026-04-23T03:54:45.754644Z","iopub.status.idle":"2026-04-23T03:54:55.427227Z","shell.execute_reply.started":"2026-04-23T03:54:45.754618Z","shell.execute_reply":"2026-04-23T03:54:55.426459Z"}}
srt_path = "/kaggle/input/datasets/liuweiq/daxiaonailong/huaqiang.srt"
results = recognize_srt_sentences(srt_path)

# %% [code] {"execution":{"iopub.status.busy":"2026-04-23T03:57:16.157011Z","iopub.execute_input":"2026-04-23T03:57:16.157706Z","iopub.status.idle":"2026-04-23T03:57:23.945530Z","shell.execute_reply.started":"2026-04-23T03:57:16.157673Z","shell.execute_reply":"2026-04-23T03:57:23.944458Z"}}
!apt install -y fonts-wqy-zenhei

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-23T03:57:26.269033Z","iopub.execute_input":"2026-04-23T03:57:26.269590Z","iopub.status.idle":"2026-04-23T03:58:02.133070Z","shell.execute_reply.started":"2026-04-23T03:57:26.269558Z","shell.execute_reply":"2026-04-23T03:58:02.132376Z"}}
video_path = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang.mp4"
output_video_path = "/kaggle/working/huaqiang_overlay_with_audio.mp4"

# 可选：传入支持中文的字体文件路径（推荐）
font_path = "/usr/sha re/fonts/truetype/wqy/wqy-zenhei.ttc"

render_video_with_results(
    video_path=video_path,
    results=results,
    output_video_path=output_video_path,
    font_path=font_path,
)