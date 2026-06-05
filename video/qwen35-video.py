# %% [markdown]
# # Qwen3.5 视频理解：刘华强买瓜
# 使用 Qwen3.5-9B 对视频进行视觉理解分析
# %% [code]

!pip install -U trl peft  datasets accelerate bitsandbytes
### 在我提了issue之后，官方修复了video token 嵌入问题，所以要从源码安装
!pip install git+https://github.com/huggingface/transformers.git@main

# %% [code]
# === 配置 ===
VIDEO_PATH = "/kaggle/input/datasets/liuweiq/daxiaonailong/liuhuaqiang-big.mp4"  # Kaggle 路径，本地测试时改为实际路径
SEGMENT_DURATION = 10  # 每段视频时长(秒)
MAX_NEW_TOKENS = 2048
OUTPUT_JSON = "/kaggle/working/video_analysis_35.json"
MAX_HISTORY_SEGMENTS = 3  # 只保留最近N段历史，防止上下文过长
TEST_SEGMENTS = None  # 只处理前N段用于测试，设为 None 处理全部
VIDEO_FPS = 2  # 视频采样帧率
VIDEO_SCALE = "640:-2"  # 缩小分辨率，-2 保证高度为偶数（H.264 要求宽高均为偶数）
MIN_SEGMENT_DURATION = 1.0  # 最少有效时长，少于此值跳过（防末尾空段）

ANALYSIS_PROMPT = """描述这段视频中的画面内容、人物动作和表情。
用中文回答。
回答不超过1000字"""

SUBTITLE_FONT_SIZE = 48
SUBTITLE_CHARS_PER_LINE = 25  # 每行最多中文字符数

# %% [markdown]
# # Step 0: 安装依赖

# %% [code]
print("=== Step 0: 安装依赖 ===")
!pip install -U trl peft transformers datasets accelerate bitsandbytes

!pip install -q qwen-vl-utils "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"

import subprocess
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
print(f"ffmpeg: {'OK' if result.returncode == 0 else 'NOT FOUND'}")

# %% [markdown]
# # Step 1: 获取视频信息并分段

# %% [code]
print("=== Step 1: 获取视频信息并分段 ===")
import json
import os

probe = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_streams", "-select_streams", "v:0", VIDEO_PATH],
    capture_output=True, text=True
)
probe_data = json.loads(probe.stdout)
vstream = probe_data["streams"][0]
width = int(vstream["width"])
height = int(vstream["height"])
fps_str = vstream.get("r_frame_rate", "30/1")
fps_num, fps_den = map(int, fps_str.split("/"))
fps = fps_num / fps_den

# 获取视频时长
duration_probe = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_format", VIDEO_PATH],
    capture_output=True, text=True
)
duration = float(json.loads(duration_probe.stdout)["format"]["duration"])

total_segments = int(duration / SEGMENT_DURATION) + (1 if duration % SEGMENT_DURATION > 0 else 0)
if TEST_SEGMENTS:
    total_segments = min(total_segments, TEST_SEGMENTS)

# 预计算每段的实际时长，过滤掉末尾时长为 0 的空段
def get_segment_actual_duration(seg_start, seg_dur, total_dur):
    actual = min(seg_start + seg_dur, total_dur) - seg_start
    return actual

print(f"Video: {width}x{height}, {fps:.2f} fps, {duration:.1f}s")
print(f"Segments: {total_segments} (each {SEGMENT_DURATION}s)")

# 用 ffmpeg 切割视频段
seg_dir = "/kaggle/working/video_segments"
os.makedirs(seg_dir, exist_ok=True)

segment_paths = []
for i in range(total_segments):
    start = i * SEGMENT_DURATION
    actual_dur = get_segment_actual_duration(start, SEGMENT_DURATION, duration)
    if actual_dur < MIN_SEGMENT_DURATION:
        print(f"  Segment {i}: skipped (only {actual_dur:.1f}s)")
        continue
    seg_path = os.path.join(seg_dir, f"seg_{i:03d}.mp4")
    ret = subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", VIDEO_PATH,
        "-t", str(actual_dur),
        "-vf", f"fps={VIDEO_FPS},scale={VIDEO_SCALE}",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an",
        seg_path
    ], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"  [ERROR] Segment {i} ffmpeg failed:\n{ret.stderr[-500:]}")
    size_mb = os.path.getsize(seg_path) / 1024 / 1024 if os.path.exists(seg_path) else 0
    segment_paths.append(seg_path)
    print(f"  Segment {i}: [{start:.0f}s - {min(start + SEGMENT_DURATION, duration):.0f}s] {size_mb:.1f} MB")

# %% [markdown]
# # Step 2: 加载 Qwen3.5 模型

# %% [code]
print("=== Step 2: 加载 Qwen3.5-9B ===")
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3.5-9B",
    trust_remote_code=True,
)

print(f"Model loaded. Device: {model.device}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# %% [markdown]
# # Step 3: 逐段分析视频

# %% [code]
print("=== Step 3: 逐段分析视频 ===")
import gc

results = []
history_texts = []

for i, seg_path in enumerate(segment_paths):
    start_sec = i * SEGMENT_DURATION
    end_sec = min((i + 1) * SEGMENT_DURATION, duration)
    actual_dur = end_sec - start_sec
    if actual_dur < MIN_SEGMENT_DURATION:
        print(f"\n--- Segment {i+1}/{total_segments} SKIP (only {actual_dur:.1f}s) ---")
        continue
    print(f"\n--- Segment {i+1}/{total_segments} [{start_sec:.0f}s - {end_sec:.0f}s] ---")

    # 防御：跳过空文件（ffmpeg 切割末尾时可能产生 0 字节文件）
    if not os.path.exists(seg_path) or os.path.getsize(seg_path) < 1024:
        print(f"  [SKIP] File missing or too small: {seg_path}")
        continue

    # 构建消息
    messages = []

    # 添加历史上下文
    if history_texts:
        for prev_text in history_texts:
            messages.append({
                "role": "assistant",
                "content": prev_text,
            })
        messages.append({
            "role": "user",
            "content": "以上是之前视频片段的描述历史，请结合上下文继续描述下一段。",
        })
        messages.append({
            "role": "assistant",
            "content": "好的，我已了解前面的画面内容，请提供下一段视频。",
        })

    # 当前视频片段
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": seg_path,
            },
            {"type": "text", "text": f"这是第{i+1}/{total_segments}段（{start_sec:.0f}s-{end_sec:.0f}s），请只描述新画面内容。\n{ANALYSIS_PROMPT}"},
        ],
    })

    # 推理：Qwen3.5 复用 Qwen3VLProcessor，直接用 apply_chat_template 处理视频
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,  # 启用思考提示，帮助模型更好地组织回答 
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.9,  # thinking mode: 0.6
            top_p=0.95,  # thinking mode: 0.95
            top_k=20,  # thinking mode: 20
            min_p=0,  # thinking mode: 0
            # presence_penalty=1.5,  # 减少重复
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 过滤思考内容：</think>之前的 ，不能写入字幕和历史
    import re
    output_text = re.sub(r"^.*?(?:<|&lt;)\s*/\s*think[^\n]*?(?:>|&gt;|\n)", "", output_text, flags=re.DOTALL | re.IGNORECASE).strip()
    results.append({
        "segment": i,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "text": output_text,
    })
    print(f"Output: {output_text}")
    history_texts.append(output_text)
    # 只保留最近N段历史
    if len(history_texts) > MAX_HISTORY_SEGMENTS:
        history_texts = history_texts[-MAX_HISTORY_SEGMENTS:]

    # 释放当前推理缓存
    del inputs, generated_ids
    torch.cuda.empty_cache()
    gc.collect()

print(f"\nAll {len(results)} segments analyzed.")

# 保存结果
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Results saved to {OUTPUT_JSON}")

# %% [markdown]
# # Step 4: 释放模型显存

# %% [code]
print("=== Step 4: 释放模型显存 ===")
del model
del processor
torch.cuda.empty_cache()
gc.collect()
print(f"GPU memory freed. CUDA allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# %% [markdown]
# # Step 5: 打印完整分析结果

# %% [code]
print("=== Step 5: 完整视频分析结果 ===\n")
for seg in results:
    print(f"{'='*60}")
    print(f"[Segment {seg['segment']+1}] {seg['start_sec']:.0f}s - {seg['end_sec']:.0f}s")
    print(f"{'='*60}")
    print(seg["text"])
    print()

# %% [markdown]
# # Step 6: 生成 ASS 字幕并烧录到视频

# %% [code]
print("=== Step 6: 生成 ASS 字幕 ===")

# 下载中文字体
font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_path = "/kaggle/working/fonts/SourceHanSansSC-Regular.otf"
os.makedirs(os.path.dirname(font_path), exist_ok=True)
if not os.path.exists(font_path):
    ret = subprocess.run(["wget", "-q", "-O", font_path, font_url], capture_output=True)
    if ret.returncode != 0:
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"], capture_output=True)
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    print(f"Font: {font_path}")
else:
    print(f"Font cached: {font_path}")

# 获取视频分辨率（用于 ASS PlayRes）
probe_v = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_streams", "-select_streams", "v:0", VIDEO_PATH],
    capture_output=True, text=True
)
video_w = int(json.loads(probe_v.stdout)["streams"][0]["width"])
video_h = int(json.loads(probe_v.stdout)["streams"][0]["height"])
print(f"Video resolution: {video_w}x{video_h}")

# 生成 ASS 字幕文件
ass_path = "/kaggle/working/subtitles.ass"
font_name = os.path.splitext(os.path.basename(font_path))[0]

def sec_to_ass(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def wrap_ass_text(text, chars_per_line=SUBTITLE_CHARS_PER_LINE):
    """中文长文本按固定字符数强制换行（ASS WrapStyle对中文无效）"""
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        while len(line) > chars_per_line:
            wrapped.append(line[:chars_per_line])
            line = line[chars_per_line:]
        wrapped.append(line)
    return "\\N".join(wrapped)

with open(ass_path, "w", encoding="utf-8") as f:
    f.write("[Script Info]\n")
    f.write("ScriptType: v4.00+\n")
    f.write(f"PlayResX: {video_w}\nPlayResY: {video_h}\n")
    f.write("WrapStyle: 1\n\n")
    f.write("[V4+ Styles]\n")
    f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
    f.write(f"Style: Default,{font_name},{SUBTITLE_FONT_SIZE},"
            f"&H00FFFF00,&H000000FF,&H00000000,&H80000000,"
            f"0,0,0,0,100,100,0,0,1,2,0,8,80,80,50,1\n\n")
    f.write("[Events]\n")
    f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    for r in results:
        start = sec_to_ass(r["start_sec"])
        end = sec_to_ass(max(r["start_sec"] + 0.1, r["end_sec"] - 0.05))
        # 先换行，再转义逗号
        text = wrap_ass_text(r["text"], chars_per_line=SUBTITLE_CHARS_PER_LINE)
        text = text.replace(",", "\\,")
        f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

print(f"ASS subtitle written: {ass_path}")

# 用 ffmpeg ass filter 直接烧录字幕
final_output = "/kaggle/working/liuhuaqiang-35-subtitled.mp4"
ret = subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_PATH,
    "-vf", f"ass={ass_path}",
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-c:a", "aac",
    final_output
], capture_output=True, text=True)

if ret.returncode != 0:
    print("ffmpeg stderr:", ret.stderr[-2000:])
else:
    final_size = os.path.getsize(final_output) / 1024 / 1024
    print(f"Final: {final_output} ({final_size:.1f} MB)")

# %% [markdown]
# # Step 7: 压缩并展示

# %% [code]
print("=== Step 7: 压缩并展示 ===")
compressed = "/kaggle/working/liuhuaqiang-35-compressed.mp4"
subprocess.run([
    "ffmpeg", "-y", "-i", final_output,
    "-vcodec", "libx264", "-crf", "28",
    compressed
], capture_output=True)
from IPython.display import Video
display(Video(compressed, embed=True))
print("=== Done ===")

# %% [markdown]

# # Step 8: 和商业化模型进行对比

# %% [markdown]

# ## Qwen3.5-Omni-Plus
https://bailian.console.aliyun.com/cn-beijing?spm=5176.42028462.nav-v2-dropdown-menu-0.d_main_2_0.fd79154aKzcKqT&tab=model&scm=20140722.M_10944401._.V_1#/efm/model_experience_center/multimodal/text-chat?modelId=qwen3.5-omni-plus
# %% [markdown]

本视频以一家便利店为背景，展开了一场充满幽默和反转的“抢劫”喜剧小短剧。以下分镜头详尽解说：

【00:00.000 – 00:02.367】画面开场为竖屏格式。镜头静止，对准天花板角落的一面凸面安防镜。镜中反射出店内主通道、货架及红色收银台。一名男子身穿黑色上衣、白色围裙正站在柜台后。此时，背景音乐响起——印度尼西亚电子舞曲（DJ remix），节拍明快，合成器音色明亮。女声吟唱一句印尼语歌词：“…ada yang baru, kalau ngana rindu, coba dengar bilang…”。店内的荧光灯营造出冷色调环境光，一切显得平静而日常。

【00:02.367 –00:05.400】 镜头切至收银员正面中景（胸以上）。他皮肤黝黑，留有短发和胡须，双手举过头顶作投降状，脸上显出无奈与顺从。背景货架上摆满Advil等药品，商品色彩杂乱。音乐节奏持续，电子底鼓每拍都强烈敲击。3. 【00:05.400 –00:09.467】同一演员突然换装成“劫匪”。他穿黑色T恤、战术背心（胸前有白字“TAKEDA”）、黑色手套，手持一把深色半自动手枪，表情由紧张转为自信微笑。镜头采用手持拍摄，轻微晃动，贴近其上半身；随后特写聚焦于枪口朝下的手枪，突出动作夸张感。音乐继续播放，无对白或环境音。4. 【00:09.467 –00:13.333】画面回切到收银员，他弯腰伸手进入收银机下方抽屉。接着，他迅速将大量一美元钞票堆放在红色柜台上，用双手聚拢整理，动作滑稽夸张。音乐重低音愈发明显，气氛愈发欢快。

【00:13.333 –00:17.867】“劫匪”再次现身，但已穿上沙色防弹衣，手里多了一个白色塑料袋。他一边挥舞袋子，一边在货架间蹦跳前进，表情欣喜若狂。头顶悬挂着红色Budweiser横幅，强化超市氛围。音乐中的女歌手重复前句歌词，电子音效点缀其间。

【00:17.867 –00:20.433】新角色登场：另一名男子身着全黑特警装备（长袖衫、长裤、战术靴），戴护目镜，手持黑色步枪。他压低身体，沿过道快速推进，动作戏剧化。镜头跟随，略带运动模糊，增加紧迫感。7. 【00:20.433 –00:22.133】收银员第三次出现，依旧高举双手，表情呆滞困惑，仿佛对眼前混乱感到不解。音乐节奏依然未停。8. 【00:22.133 –00:28.167】劫匪回到柜台前，双手抓着大把现金，满脸得意。突然门口灯光变暗，一个穿着肌肉质感灰色蝙蝠侠战衣、斗篷飘动的人物缓缓步入。劫匪的笑容瞬间消失，表情转为震惊。9. 【00:28.167 –00:33.933】 劫匪放下钞票，双手摊开做出“这是什么情况”的动作。镜头快速切到沙色防弹衣劫匪和黑衣特警，两人各自露出惊讶表情，仿佛被突如其来的超级英雄震慑。10. 【00:33.933 – 00:37.200】劫匪与蝙蝠侠对峙。蝙蝠侠背对镜头，劫匪则手足无措地摆动双手，试图解释。镜头采用手持跟拍，略有不稳。此时，音乐中加入一段男声说唱：“Ada di Papua, ada juga di Irian Jaya.”，节奏鲜明。

【00:37.200 –00:44.367】 收银员忽然面带灿烂笑容，举起双手挥动，仿佛欢迎英雄到来。紧接着，镜头近距离捕捉蝙蝠侠的手部细节——黑色手套紧握拳头，金色腰带显眼。12. 【00:44.367 –00:50.333】劫匪脸部特写，先是一副茫然，随即眉头紧锁，咬牙愤怒。镜头再转到蝙蝠侠，其头盔双眼发出红光，气氛骤然紧张。

【00:50.333 – 00:53.567】门口再度亮起炫目红光，另一位身穿蓝红相间紧身衣、肩部饰有金色星星的角色走入店内，双眼也闪烁红光。劫匪见状立刻抱头蹲下，表现出彻底崩溃。14. 【00:53.567 –01:02.033】镜头特写新角色面部：眼神凶狠，红光照耀。接着快速剪辑之前片段——劫匪持枪、收银员数钱、沙色防弹衣劫匪跳舞——形成快节奏蒙太奇，配乐高潮部分反复吟唱“Coba dengar bilang”，加强戏谑感。

【01:02.033 –01:21.533】 最后段落将之前所有关键动作循环播放：收银员疯狂撒钞票，两个劫匪交替出现，特警奔跑，收银员举手，劫匪拿钱，最终定格在他低头看钞票的画面。音乐渐弱直至结束，整个视频以喜剧方式收场，荒诞又带点讽刺。总的来说，这段短视频通过快速剪辑、角色扮演、流行电子乐和夸张表演，营造出一场毫无逻辑却极具娱乐性的“抢劫大战”，融合了网络模因文化与超级英雄元素，使观众在笑声中体验意外反转的快感。
# %% [markdown]

# ## Gemini-3.1-Pro 网页版

# %% [markdown]

🏪 午夜便利店的“奇妙”抢劫案
伟清，让我们切入第一视角，看看这段令人啼笑皆非的便利店监控录像。准备好，这绝对不是标准意义上的警匪悬疑片。

🕺 第一幕：踩着节拍的“悍匪”
动感登场：伴随着一首极其洗脑、节奏欢快的印尼神曲《Ngana Rindu》，画面中闯入了几个身穿黑色战术背心、装备齐全的劫匪。

匪夷所思的步伐：他们没有大喊大叫，也没有暴力威吓，而是完全踩着音乐的节拍，在货架间摇曳生姿！他们一边扫荡着货架上的商品，一边保持着极强的身体律动，仿佛便利店一秒变成了午夜派对的舞池。

💸 第二幕：过度配合的收银员
绝望的散财童子：镜头切给便利店的收银员小哥。面对这场“歌舞青春版”抢劫，他表现出了极其浮夸的惊恐。

主动出击：他满脸生无可恋，手上的动作却毫不含糊，直接从收银机里抓起一把又一把的钞票，疯狂且主动地往劫匪手里塞。那表情配合着肢体动作，仿佛在说：“钱都拿走，求你们别跳了！”

🦇 第三幕：高能反转与天降神兵
零钞的困惑：带头的劫匪拿着厚厚一沓美钞，开始得意地清点。但他很快发现事情不对——怎么清一色全是一美元的零钞？他抬起头，露出了大大的疑惑。

正义降临：就在劫匪怀疑人生之际，便利店的大门赫然出现了一个充满压迫感的身影——蝙蝠侠！他站在门口扭动脖子，按得指关节咔咔作响，随时准备暴力执法。

深藏功与名：此时，刚刚还满脸惊恐的收银员小哥瞬间切换状态，对着镜头淡定地比了一个“OK”的手势，仿佛一切尽在掌握。

绝望加倍：正当劫匪已经不知所措时，镜头一闪，门口的“救兵”竟然又换成了双眼泛着致命红光的祖国人（Homelander）！劫匪原本疑惑的表情彻底凝固成了绝望。

解说总结：
这段视频完全是一场荒诞的视听混搭。用最欢快的音乐，演最严肃的抢劫，最后用最意想不到的超级英雄进行降维打击。伟清，这剧情的脑洞和反转，绝对把短视频的娱乐精神发挥到了极致。
