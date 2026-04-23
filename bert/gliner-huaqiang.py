# %% [code] {"jupyter":{"outputs_hidden":false}}
from IPython.display import clear_output
!pip install gliner2 -U
!pip install srt
clear_output()

# %% [code] {"execution":{"iopub.status.busy":"2026-04-23T03:42:26.546464Z","iopub.execute_input":"2026-04-23T03:42:26.547163Z","iopub.status.idle":"2026-04-23T03:42:32.097145Z","shell.execute_reply.started":"2026-04-23T03:42:26.547134Z","shell.execute_reply":"2026-04-23T03:42:32.096246Z"}}
from pathlib import Path
import srt
import warnings
warnings.filterwarnings("ignore")

from gliner2 import GLiNER2

# ===================== 【修复1】只加载一次模型 =====================
extractor = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-23T03:45:40.348093Z","iopub.execute_input":"2026-04-23T03:45:40.348416Z","iopub.status.idle":"2026-04-23T03:45:41.275458Z","shell.execute_reply.started":"2026-04-23T03:45:40.348389Z","shell.execute_reply":"2026-04-23T03:45:41.274827Z"}}
from transformers import pipeline
sentiment_model  = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
# emotion_model = pipeline(
#     "text-classification",
#     model="bardsai/chinese-emotion-classification",
#     device=0
# )

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-23T03:49:36.387743Z","iopub.execute_input":"2026-04-23T03:49:36.388421Z","iopub.status.idle":"2026-04-23T03:49:36.396600Z","shell.execute_reply.started":"2026-04-23T03:49:36.388392Z","shell.execute_reply":"2026-04-23T03:49:36.396032Z"}}
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

        # ==================== 新加：调用精准情感模型 ====================
        # real_emotion = sentiment_model(sentence)[0]

        item = {
            "index": sub.index,
            "start": format_timedelta_srt(sub.start),
            "end": format_timedelta_srt(sub.end),
            "text": sentence,
            "entities": result.get("entities", {}),
            "sentiment": result.get("情感", "中性"),
            # "real_emotion": real_emotion  # 把精准情感也存起来
        }
        all_results.append(item)

        # 输出（完全保留你原来的格式，只多加一行）
        print(f"【{sub.index}】{item['start']} --> {item['end']}")
        print(f"内容：{sentence}")
        print(f"实体：{item['entities']}")
        print(f"GLiNER情感：{item['sentiment']}")       # 原来的
        print(f"✅ 真实情感：{item['real_emotion']}")    # 新加的（准）
        print("-" * 60)

    return all_results

# %% [code] {"execution":{"iopub.status.busy":"2026-04-23T03:48:12.056693Z","iopub.execute_input":"2026-04-23T03:48:12.057232Z","iopub.status.idle":"2026-04-23T03:48:13.023944Z","shell.execute_reply.started":"2026-04-23T03:48:12.057186Z","shell.execute_reply":"2026-04-23T03:48:13.023122Z"}}
from transformers import pipeline

sentiment = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
print(sentiment("这个AI产品体验非常差"))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-23T03:49:40.324192Z","iopub.execute_input":"2026-04-23T03:49:40.324633Z","iopub.status.idle":"2026-04-23T03:49:48.250391Z","shell.execute_reply.started":"2026-04-23T03:49:40.324607Z","shell.execute_reply":"2026-04-23T03:49:48.249470Z"}}
srt_path = "/kaggle/input/datasets/liuweiq/daxiaonailong/huaqiang.srt"
results = recognize_srt_sentences(srt_path)