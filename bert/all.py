# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-26T05:37:52.023006Z","iopub.execute_input":"2026-02-26T05:37:52.023219Z","iopub.status.idle":"2026-02-26T05:38:21.715956Z","shell.execute_reply.started":"2026-02-26T05:37:52.023197Z","shell.execute_reply":"2026-02-26T05:38:21.715315Z"}}
"""
BERT完形填空测试公共函数库
提供模型加载、评分、自回归答题等通用功能
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def setup_device() -> torch.device:
    """
    设置计算设备（GPU或CPU）
    
    Returns:
        torch.device: 可用的计算设备
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")
    return device


def load_model(model_name: str = "roberta-base", device: Optional[torch.device] = None) -> Tuple:
    """
    加载预训练的MLM模型和分词器
    
    Args:
        model_name: 模型名称，默认为 "roberta-base"
        device: 计算设备，如果为None则自动选择
        
    Returns:
        tuple: (tokenizer, model, device)
    """
    if device is None:
        device = setup_device()
    
    print(f"正在加载 {model_name} 到显存...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    
    return tokenizer, model, device


def score_sentence(
    sentence: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 512
) -> float:
    """
    计算句子的不自然度（loss值）- 旧方法，用于向后兼容
    
    Args:
        sentence: 待评分的句子
        tokenizer: 分词器
        model: MLM模型
        device: 计算设备
        max_length: 最大token长度
        
    Returns:
        float: loss值（越低越好）
    """
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    inputs["labels"] = inputs["input_ids"].clone()
    
    # 将数据送入设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        loss = outputs.loss.item()
    
    return loss


def score_candidate_word(
    text_with_mask: str,
    candidate_word: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 512,
    stride: int = 128
) -> float:
    """
    使用mask token计算候选词的精确损失（推荐方法）
    支持单词和多词短语，使用滑动窗口自动处理长文本
    
    Args:
        text_with_mask: 包含mask token的句子
        candidate_word: 候选词或短语
        tokenizer: 分词器
        model: MLM模型
        device: 计算设备
        max_length: 最大token长度
        stride: 滑动窗口步长
        
    Returns:
        float: loss值（越低越好）
    """
    # 1. 先tokenize候选词，看看它被分成多少个token
    candidate_tokens = tokenizer.tokenize(candidate_word)
    candidate_ids = tokenizer.convert_tokens_to_ids(candidate_tokens)
    
    # 2. 根据候选词的token数量，创建对应数量的mask
    if len(candidate_tokens) == 1:
        # 单个token的情况，使用精确的mask方法
        inputs = tokenizer(
            text_with_mask,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding=True
        )
        
        # 找到包含mask token的那个chunk
        mask_token_id = tokenizer.mask_token_id
        chunk_idx = 0
        
        # 如果有多个chunks，找到包含mask的那个
        if "overflow_to_sample_mapping" in inputs:
            for idx in range(len(inputs["input_ids"])):
                if mask_token_id in inputs["input_ids"][idx]:
                    chunk_idx = idx
                    break
        
        # 构造labels：只在mask位置计算loss
        labels = torch.full_like(inputs["input_ids"], fill_value=-100)
        mask_positions = (inputs["input_ids"][chunk_idx] == mask_token_id).nonzero(as_tuple=True)[0]
        
        if len(mask_positions) > 0:
            labels[chunk_idx, mask_positions[0]] = candidate_ids[0]
        
        # 只使用包含mask的那个chunk
        selected_inputs = {
            "input_ids": inputs["input_ids"][chunk_idx:chunk_idx+1],
            "attention_mask": inputs["attention_mask"][chunk_idx:chunk_idx+1],
            "labels": labels[chunk_idx:chunk_idx+1]
        }
        
    else:
        # 多个token的情况，用候选词替换mask，计算整体loss
        text_with_candidate = text_with_mask.replace(tokenizer.mask_token, candidate_word)
        inputs = tokenizer(
            text_with_candidate,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding=True
        )
        
        # 对于多词短语，使用第一个chunk（通常候选词在开始部分）
        selected_inputs = {
            "input_ids": inputs["input_ids"][0:1],
            "attention_mask": inputs["attention_mask"][0:1],
            "labels": inputs["input_ids"][0:1].clone()
        }
    
    # 将数据送入设备
    selected_inputs = {k: v.to(device) for k, v in selected_inputs.items()}
    
    # 计算loss
    with torch.no_grad():
        outputs = model(**selected_inputs)
        return outputs.loss.item()


def clean_text(text: str, target_marker: str, option: str) -> str:
    """
    清理和替换文本中的标记
    
    Args:
        text: 原始文本
        target_marker: 目标标记（如 "__1__"）
        option: 要填入的选项
        
    Returns:
        str: 清理后的文本
    """
    # 替换目标标记
    current_test_text = text.replace(target_marker, option)
    # 将其他题号标记替换为占位符
    clean_test_text = re.sub(r"__\d+__", "___", current_test_text)
    return clean_test_text


def crop_context(
    text: str,
    marker: str,
    context_size: int = 800
) -> str:
    """
    裁剪文本，保留目标标记周围的上下文
    
    Args:
        text: 完整文本
        marker: 目标标记
        context_size: 上下文大小（前后各取的字符数）
        
    Returns:
        str: 裁剪后的文本
    """
    marker_pos = text.find(marker)
    if marker_pos == -1:
        return text
    
    start_pos = max(0, marker_pos - context_size)
    end_pos = min(len(text), marker_pos + context_size)
    return text[start_pos:end_pos]


def autoregressive_cloze_test(
    raw_text: str,
    options_dict: Dict[int, List[str]],
    tokenizer,
    model,
    device: torch.device,
    start_idx: int = 1,
    end_idx: int = 20,
    context_size: int = 800,  # 保留参数以向后兼容，但不再使用
    max_length: int = 512,
    letters: Optional[List[str]] = None
) -> Dict[int, Dict]:
    """
    自回归完形填空答题
    使用tokenizer的滑动窗口自动处理长文本
    
    Args:
        raw_text: 原始文本（包含 __n__ 标记）
        options_dict: 选项字典 {题号: [选项列表]}
        tokenizer: 分词器
        model: MLM模型
        device: 计算设备
        start_idx: 起始题号
        end_idx: 结束题号（不包含）
        context_size: （已弃用）保留以向后兼容，tokenizer自动处理滑动窗口
        max_length: 最大token长度
        letters: 选项字母列表，默认为 ["A", "B", "C", "D"]
        
    Returns:
        dict: {题号: {"option": 选项, "letter": 字母, "loss": loss值}}
    """
    if letters is None:
        letters = ["A", "B", "C", "D"]
    
    print("\n--- 开始带 GPU 加速的自回归答题 ---")
    
    working_text = raw_text
    results_summary = {}
    
    for i in range(start_idx, end_idx + 1):
        if i not in options_dict:
            continue
            
        target_marker = f"__{i}__"
        results = {}
        
        # 将目标标记替换为 mask token
        text_with_mask = working_text.replace(target_marker, tokenizer.mask_token)
        # 将其他题号标记替换为占位符
        clean_text_with_mask = re.sub(r"__\d+__", "___", text_with_mask)
        
        # 不再需要手动裁剪，tokenizer会自动用滑动窗口处理长文本
        for opt in options_dict[i]:
            # 使用新的基于mask的评分方法（自带滑动窗口）
            loss = score_candidate_word(
                clean_text_with_mask,
                opt,
                tokenizer,
                model,
                device,
                max_length
            )
            
            results[opt] = loss
        
        # 找出 Loss 最低的选项
        best_opt = min(results, key=results.get)
        best_loss = results[best_opt]
        best_idx = options_dict[i].index(best_opt)
        best_letter = letters[best_idx]
        
        # 填入答案，实现自回归
        working_text = working_text.replace(target_marker, best_opt)
        
        # 保存结果
        results_summary[i] = {
            "option": best_opt,
            "letter": best_letter,
            "loss": best_loss,
            "all_scores": results
        }
        
        print(
            f"第 {i:02d} 题 -> 模型选择: {best_letter}. {best_opt} (Loss: {best_loss:.4f})"
        )
    
    print("\n--- 答题结束 ---")
    return results_summary


def simple_cloze_test(
    prompt_template: str,
    options: List[str],
    tokenizer,
    model,
    device: torch.device
) -> Dict[str, float]:
    """
    简单的完形填空测试（单题）
    
    Args:
        prompt_template: 提示模板，使用 {} 作为占位符
        options: 选项列表
        tokenizer: 分词器
        model: MLM模型
        device: 计算设备
        
    Returns:
        dict: {选项: loss值}，按loss从小到大排序
    """
    print("\n--- 开始免训练(Zero-shot)打分 ---")
    
    results = {}
    
    for opt in options:
        complete_sentence = prompt_template.format(opt)
        loss_score = score_sentence(complete_sentence, tokenizer, model, device)
        results[opt] = loss_score
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    for rank, (word, loss) in enumerate(sorted_results, 1):
        if rank == 1:
            print(f"🏆 最佳选项 -> {word} (不自然度 Loss: {loss:.4f})")
        else:
            print(f"   淘汰选项 -> {word} (不自然度 Loss: {loss:.4f})")
    
    return dict(sorted_results)


def print_results_table(
    results: Dict[int, Dict],
    answers: Optional[Dict[int, str]] = None
) -> None:
    """
    打印答题结果表格
    
    Args:
        results: autoregressive_cloze_test 返回的结果
        answers: 标准答案字典 {题号: "字母"}
    """
    print("\n| 题号 | 模型选择 | 标准答案 | 批改 |")
    print("|---:|:---|:---|:---:|")
    
    for i in sorted(results.keys()):
        model_choice = f"{results[i]['letter']}. {results[i]['option']}"
        
        if answers and i in answers:
            is_correct = results[i]['letter'] == answers[i]
            mark = "✅" if is_correct else "❌"
            print(f"| {i:02d} | {model_choice} | {answers[i]} | {mark} |")
        else:
            print(f"| {i:02d} | {model_choice} | - | - |")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 考研英语到底有多变态？BERT只答对一半

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## BERT 简单题目测试

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T05:38:21.717368Z","iopub.execute_input":"2026-02-26T05:38:21.717762Z","iopub.status.idle":"2026-02-26T05:38:31.230132Z","shell.execute_reply.started":"2026-02-26T05:38:21.717737Z","shell.execute_reply":"2026-02-26T05:38:31.229262Z"},"jupyter":{"outputs_hidden":false}}
# from bert_utils import load_model, simple_cloze_test

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 题目和多 Token 选项
prompt_template = (
    "Despite the {} evidence, the jury found it difficult to reach a unanimous verdict."
)
options = ["overwhelming", "vague", "insufficient", "unreliable"]

# 执行简单测试
results = simple_cloze_test(prompt_template, options, tokenizer, model, device)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## BERT 2022年考研英语一完形填空

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T05:38:31.231085Z","iopub.execute_input":"2026-02-26T05:38:31.231309Z","iopub.status.idle":"2026-02-26T05:38:34.644040Z","shell.execute_reply.started":"2026-02-26T05:38:31.231288Z","shell.execute_reply":"2026-02-26T05:38:34.643345Z"},"jupyter":{"outputs_hidden":false}}
#from bert_utils import autoregressive_cloze_test, load_model

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 原始文本
raw_text = """
The idea that plants have some degree of consciousness first took root in the early 2000s; the term “plant neurobiology” was __1__ around the notion that some aspects of plant behavior could be __2__ to intelligence in animals. __3__ plants lack brains, the firing of electrical signals in their stems and leaves nonetheless triggered responses that __4__ consciousness, researchers previously reported.

But such an idea is untrue, according to a new opinion article. Plant biology is complex and fascinating, but it __5__ so greatly from that of animals that so-called __6__ of plants’ intelligence is inconclusive, the authors wrote.

Beginning in 2006, some scientists have __7__ that plants possess neuron-like cells that interact with hormones and neurotransmitters, __8__ “a plant nervous system, __9__ to that in animals,” said lead study author Lincoln Taiz, “They __10__ claimed that plants have “brain-like command centers” at their root tips.”

This __11__ makes sense if you simplify the workings of a complex brain, __12__ it to an array of electrical pulses; cells in plants also communicate through electrical signals. __13__, the signaling in a plant is only __14__ similar to the firing in a complex animal brain, which is more than “a mass of cells that communicate by electricity,” Taiz said.

“For consciousness to evolve, a brain with a threshold __15__ of complexity and capacity is required,” he __16__.”Since plants don’t have nervous systems, the __17__ that they have consciousness are effectively zero.”

And what’s so great about consciousness, anyway? Plants can’t run away from __18__, so investing energy in a body system which __19__ a threat and can feel pain would be a very __20__ evolutionary strategy, according to the article.
"""

# 选项字典
options_dict = {
    1: ["coined", "discovered", "collected", "issued"],
    2: ["attributed", "directed", "compared", "confined"],
    3: ["unless", "when", "once", "though"],
    4: ["coped with", "consisted of", "hinted at", "extended"],
    5: ["suffers", "benefits", "develops", "differs"],
    6: ["acceptance", "evidence", "cultivation", "creation"],
    7: ["doubted", "denied", "argued", "requested"],
    8: ["adapting", "forming", "repairing", "testing"],
    9: ["analogous", "essential", "suitable", "sensitive"],
    10: ["just", "ever", "still", "even"],
    11: ["restriction", "experiment", "perspective", "demand"],
    12: ["attaching", "reducing", "returning", "exposing"],
    13: ["However", "Moreover", "Therefore", "Otherwise"],
    14: ["temporarily", "literally", "superficially", "imaginarily"],
    15: ["list", "level", "label", "local"],
    16: ["recalled", "agreed", "questioned", "added"],
    17: ["chances", "risks", "excuses", "assumptions"],
    18: ["danger", "failure", "warning", "control"],
    19: ["represents", "includes", "reveals", "recognizes"],
    20: ["humble", "poor", "practical", "easy"],
}

# 执行自回归答题
results1 = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=1, end_idx=20
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# | 题号 | 模型选择 | 标准答案 | 批改 |
# |---:|:---|:---|:---:|
# | 01 | A. coined | A. coined | ✅ |
# | 02 | C. compared | C. compared | ✅ |
# | 03 | D. though | D. though | ✅ |
# | 04 | A. coped with | C. hinted at | ❌ |
# | 05 | A. suffers | D. differs | ❌ |
# | 06 | B. evidence | B. evidence | ✅ |
# | 07 | B. denied | C. argued | ❌ |
# | 08 | B. forming | B. forming | ✅ |
# | 09 | A. analogous | A. analogous | ✅ |
# | 10 | A. just | D. even | ❌ |
# | 11 | B. experiment | C. perspective | ❌ |
# | 12 | A. attaching | B. reducing | ❌ |
# | 13 | C. Therefore | A. However | ❌ |
# | 14 | C. superficially | C. superficially | ✅ |
# | 15 | B. level | B. level | ✅ |
# | 16 | B. agreed | D. added | ❌ |
# | 17 | D. assumptions | A. chances | ❌ |
# | 18 | A. danger | A. danger | ✅ |
# | 19 | A. represents | D. recognizes | ❌ |
# | 20 | B. poor | B. poor | ✅ |

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## BERT 2023年考研英语一完形填空

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T05:38:34.645176Z","iopub.execute_input":"2026-02-26T05:38:34.645637Z","iopub.status.idle":"2026-02-26T05:38:38.021953Z","shell.execute_reply.started":"2026-02-26T05:38:34.645586Z","shell.execute_reply":"2026-02-26T05:38:38.021276Z"},"jupyter":{"outputs_hidden":false}}
#from bert_utils import autoregressive_cloze_test, load_model

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 修复 OCR 乱码后的纯净文本
raw_text = """
Caravanserais were roadside inns that were built along the Silk Road in areas including China, North Africa and the Middle East. They were typically __1__ outside the walls of a city or village and were usually funded by governments or __2__. This word “Caravanserais” is a __3__ of the Persian word “karvan”, which means a group of travellers or a caravan, and seray, a palace or enclosed building. The term caravan was used to __4__ groups of people who travelled together across the ancient network for safety reasons, __5__ merchants, travellers or pilgrims. From the 10th century onwards, as merchant and travel routes become more developed, the __6__ of the Caravanserais increased and they served as a safe place for people to rest at night. Travellers on the Silk Road __7__ the possibility of being attacked by thieves or being __8__ to extreme conditions. For this reason, Caravanserais were strategically placed __9__ they could be reached in a day’s travel time. Caravanserais served as an informal __10__ point for the various people who travelled the Silk Road. __11__, those structures became important centers for culture __12__ and interaction, with travelers sharing their cultures, ideas and beliefs, __13__ taking knowledge with them, greatly __14__ the development of several civilizations. Caravanserais were also an important marketplace for commodities and __15__ in the trade of goods along the Silk Road. __16__, it was frequently the first stop for merchants looking to sell their wares and __17__ supplies for their own journeys. It is __18__ that around 12,000 to 15,000 caravanserais were built along the Silk Road, __19__ only about 3,000 are known to remain today, many of which are in __20__.
"""

# 选项字典
options_dict = {
    1: ["displayed", "occupied", "located", "equipped"],
    2: ["privately", "regularly", "respectively", "permanently"],
    3: ["definition", "transition", "substitution", "combination"],
    4: ["classify", "record", "describe", "connect"],
    5: ["apart from", "instead of", "such as", "along with"],
    6: ["construction", "restoration", "impression", "evaluation"],
    7: ["doubted", "faced", "accepted", "reduced"],
    8: ["assigned", "subjected", "accustomed", "opposed"],
    9: ["so that", "even if", "now that", "in case"],
    10: ["talking", "starting", "breaking", "meeting"],
    11: ["By the way", "On occasion", "In comparison", "As a result"],
    12: ["heritage", "revival", "exchange", "status"],
    13: ["with regard to", "in spite of", "as well as", "in line with"],
    14: ["completing", "influencing", "resuming", "pioneering"],
    15: ["aided", "invested", "failed", "competed"],
    16: ["Rather", "Indeed", "Otherwise", "However"],
    17: ["go in for", "stand up for", "close in on", "stock up on"],
    18: ["believed", "predicted", "recalled", "implied"],
    19: ["until", "because", "unless", "although"],
    20: ["ruins", "debt", "fashion", "series"],
}

# 执行自回归答题
results2 = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=1, end_idx=20
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# | 题号 | 模型选择 | 标准答案 | 批改 |
# |---:|:---|:---|:---:|
# | 01 | C. located | C. located | ✅ |
# | 02 | A. privately | A. privately | ✅ |
# | 03 | A. definition | D. combination | ❌ |
# | 04 | C. describe | C. describe | ✅ |
# | 05 | B. instead of | C. such as | ❌ |
# | 06 | A. construction | A. construction | ✅ |
# | 07 | D. reduced | B. faced | ❌ |
# | 08 | B. subjected | B. subjected | ✅ |
# | 08 | B. predicted | A. believed | ❌ |
# | 09 | B. even if | A. so that | ❌ |
# | 10 | B. starting | D. meeting | ❌ |
# | 11 | A. By the way | D. As a result | ❌ |
# | 12 | B. revival | C. exchange | ❌ |
# | 13 | A. with regard to | C. as well as | ❌ |
# | 14 | C. resuming | B. influencing | ❌ |
# | 15 | A. aided | A. aided | ✅ |
# | 16 | D. However | B. Indeed | ❌ |
# | 17 | A. go in for | D. stock up on | ❌ |
# | 19 | D. although | D. although | ✅ |
# | 20 | A. ruins | A. ruins | ✅ |

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## BERT 2019年上海英语高考完形填空

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T05:38:38.022882Z","iopub.execute_input":"2026-02-26T05:38:38.023117Z","iopub.status.idle":"2026-02-26T05:38:41.531809Z","shell.execute_reply.started":"2026-02-26T05:38:38.023094Z","shell.execute_reply":"2026-02-26T05:38:41.530995Z"},"jupyter":{"outputs_hidden":false}}
#from bert_utils import autoregressive_cloze_test, load_model

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 纯净原文本 (1-15题)
raw_text = """
We're told that writing is dying. Typing on keyboards and screens __1__ written communication today. Learning cursive, joined-up handwriting was once __2__ in schools. But now, not so much. Countries such as Finland have dropped joined-up handwriting lessons in __3__ of typing courses. And in the US, the requirement to learn cursive has been left out of core standards since 2013. A few US states still place value on formative cursive education, such as Arizona, but they're not the __4__.

Some experts point out that writing lessons can have indirect __5__. Anne Trubek, author of The History and Uncertain Future of Handwriting, argues that such lessons can reinforce a skill called automaticity. That's when you've perfected a task, and can do it almost without thinking, __6__ you extra mental bandwidth to think about other things while you're doing the task. In this sense, Trubek likens handwriting to __7__.

"Once you have driven for a while, you don't __8__ think 'Step on gas now' [or] 'Turn the steering wheel a bit'," she explains. "You just do it. That's what we want children to __9__ when learning to write. You don't think 'now make a loop going up for the 't'' or 'now look for the letter 'r' on the keyboard'."

Trubek has written many essays and books on handwriting, and she doesn't believe it will die out for a very long time, "ever", but she believes students are learning how to type faster without looking at the keys at __10__ ages, and students are learning automaticity with keyboards that was once exclusive to handwriting: to type faster than they could write, granting them extra time to think about word choice or sentence structure. In a piece penned for the New York Times last year, Trubek argued that due to the improved automaticity of keyboards, today's children may well become better communicators in text, as __11__ take up less of their education. 

This is a(n) __12__ that has attracted both criticism and support. She explains that two of the most common arguments she hears from detractors regarding the decline of handwriting is that not __13__ it will result in a loss of history and a "loss of personal touch".

On the former she __14__ that 95% of handwritten manuscripts can't be read by the average person anyway – "that's why we have paleographers," she explains, paleography being the study of ancient styles of writing – while the latter refers to the warm __15__ we give to handwritten personal notes, such as thank-you cards.
"""

# 选项字典 (1-15题)
options_dict = {
    1: ["abandons", "dominates", "enters", "absorbs"],
    2: ["compulsory", "opposite", "crucial", "relevant"],
    3: ["in want of", "in case of", "in favour of", "in addition to"],
    4: ["quantity", "minimum", "quality", "majority"],
    5: ["responsibility", "benefits", "resources", "structure"],
    6: ["granting", "getting", "bringing", "costing"],
    7: ["sleeping", "driving", "reviewing", "operating"],
    8: ["eventually", "constantly", "frequently", "consciously"],
    9: ["adopt", "reach", "acquire", "activate"],
    10: ["slower", "later", "faster", "earlier"],
    11: ["handwriting", "typing", "reading", "spelling"],
    12: ["trust", "book", "view", "smile"],
    13: ["containing", "spreading", "choosing", "preserving"],
    14: ["commits", "counters", "completes", "composes"],
    15: ["associations", "resources", "procedures", "interactions"],
}

# 执行自回归答题
results3 = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=1, end_idx=15
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# | 题号 | 模型选择 | 标准答案 | 批改 | 题号 | 模型选择 | 标准答案 | 批改 |
# |---|---|---|---|---|---|---|---|
# | 01 | A. abandons | B. dominates | ❌ | 09 | A. adopt | C. acquire | ❌ |
# | 02 | C. crucial | A. compulsory | ❌ | 10 | D. earlier | D. earlier | ✅ |
# | 03 | C. in favour of | C. in favour of | ✅ | 11 | A. handwriting | A. handwriting | ✅ |
# | 04 | B. minimum | D. majority | ❌ | 12 | C. view | C. view | ✅ |
# | 05 | D. structure | B. benefits | ❌ | 13 | D. preserving | D. preserving | ✅ |
# | 06 | A. granting | A. granting | ✅ | 14 | D. composes | B. counters | ❌ |
# | 07 | B. driving | B. driving | ✅ | 15 | A. associations | A. associations | ✅ |
# | 08 | D. consciously | D. consciously | ✅ |  |  |  |  |

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## BERT 2019年上海英语春考完形填空

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T05:38:41.533835Z","iopub.execute_input":"2026-02-26T05:38:41.534155Z","iopub.status.idle":"2026-02-26T05:38:44.815548Z","shell.execute_reply.started":"2026-02-26T05:38:41.534130Z","shell.execute_reply":"2026-02-26T05:38:44.814758Z"},"jupyter":{"outputs_hidden":false}}
#from bert_utils import autoregressive_cloze_test, load_model

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 纯人工精校录入的原文
raw_text = """
More people are travelling than ever before, and lower barriers to entry and falling costs means they are doing so for __41__ periods.

The rise of "city breaks" 48-hour bursts of foreign cultures, easier on the pocket and annual leave balance has increased tourist numbers, but not their __42__ spread. The same attractions have been used to market cities such as Paris, Barcelona and Venice for decades, and visitors use the same infrastructure as residents to reach them. "Too many people do the same thing at the exact same time," says Font. "For __43__, the city no longer belongs to them."

This starts with marketing, says Font, who notes that Amsterdam has started advising visitors to seek __44__ outside of the city center on its official website. "That takes some balls, really, to do that. But only so many people will look at the website, and it means they can say to their residents they're doing all they can (to ease congestion)."

But it also __45__ a better way, it is calling "de-tourism": sustainable travel tips and __46__ itineraries for exploring an authentic Venice, off the paths beaten by the 28 million visitors who flock there each year.

A greater variety of __47__ for prospective visitors—ideas for what to do in off-peak seasons, for example, or outside of the city center—can have the effect of diverting them from already saturated landmarks, or __48__ short breaks away in the first place.

Longer stays __49__ the pressure, says Font. "If you go to Paris for two days, you're not going to go to the Eiffel Tower. If you go for two weeks, you're not going to go to the Eiffel tower 14 times."

Similarly, repeat visitors have a better sense of the __50__, "We should be asking how do we get tourists to __51__, not how to get them to come for the first time. If they're coming for the fifth time, it is much easier to integrate their behavior with ours."

Local governments can foster this sustainable activity by giving preference to responsible operators and even high-paying consumers. Font says cities could stand to be more selective about the tourists they try to attract when the current metric for marketing success is how many there are, and how far they've come. "You're thinking, 'yeah but at what cost...'"

He points to unpublished data from the Barcelona Tourist Board that prioritizes Japanese tourists for spending an average of 640 more per day than French tourists—a(n) __52__ that fails to take into account their bigger carbon footprint. __53__ tourists are also more likely to be repeat visitors that come at off-peak times, buy local product, and __54__ less crowded parts of the city—all productive steps towards more __55__ and more peaceful relations with residents.
"""

# 选项字典 (41 - 55题)
options_dict = {
    41: ["longer", "shorter", "wider", "clearer"],
    42: ["environmental", "national", "economic", "geographic"],
    43: ["locals", "tourists", "visitors", "cleaners"],
    44: ["transports", "accommodation", "restaurants", "service"],
    45: ["addresses", "introduces", "proposes", "receives"],
    46: ["separate", "individual", "alternative", "objective"],
    47: ["reform", "guidance", "invitation", "support"],
    48: ["convincing", "discouraging", "preventing", "resisting"],
    49: ["peace", "risk", "leisure", "ease"],
    50: ["culture", "knowledge", "entertainment", "ability"],
    51: ["go with", "bring up", "come back", "lay off"],
    52: ["distinction", "harmony", "association", "comparison"],
    53: ["French", "Italian", "Spanish", "German"],
    54: ["carry out", "give into", "spread out", "impact on"],
    55: ["sight", "complex", "temporary", "sustainable"],
}

# 执行自回归答题
results4 = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=41, end_idx=55
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# | 题号 | 模型选择 | 标准答案 | 批改 | 题号 | 模型选择 | 标准答案 | 批改 |
# |---|---|---|---|---|---|---|---|
# | 41 | B. shorter | B. shorter | ✅ | 49 | D. ease | D. ease | ✅ |
# | 42 | B. national | D. geographic | ❌ | 50 | B. knowledge | A. culture | ❌ |
# | 43 | A. locals | A. locals | ✅ | 51 | C. come back | C. come back | ✅ |
# | 44 | B. accommodation | B. accommodation | ✅ | 52 | A. distinction | D. comparison | ❌ |
# | 45 | B. introduces | C. proposes | ❌ | 53 | C. Spanish | A. French | ❌ |
# | 46 | C. alternative | C. alternative | ✅ | 54 | C. spread out | C. spread out | ✅ |
# | 47 | B. guidance | B. guidance | ✅ | 55 | B. complex | D. sustainable | ❌ |
# | 48 | C. preventing | B. discouraging | ❌ |  |  |  |  |

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 准确率评估

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-26T05:38:44.816512Z","iopub.execute_input":"2026-02-26T05:38:44.816785Z","iopub.status.idle":"2026-02-26T05:38:44.822089Z","shell.execute_reply.started":"2026-02-26T05:38:44.816761Z","shell.execute_reply":"2026-02-26T05:38:44.821266Z"}}
def calculate_accuracy(results, correct_answers):
    """
    计算答题准确率
    
    参数:
        results: autoregressive_cloze_test 返回的结果字典 {题号: {"letter": "A", ...}}
        correct_answers: 字典，键为题号，值为正确答案的索引(0=A, 1=B, 2=C, 3=D)
    
    返回:
        accuracy: 准确率 (0-1之间的浮点数)
        correct_count: 正确题目数
        total_count: 总题目数
    """
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    correct_count = 0
    total_count = len(results)
    
    for question_num, result_dict in results.items():
        if question_num in correct_answers:
            # 从 letter 转换为索引
            selected_letter = result_dict['letter']
            selected_idx = letter_to_idx[selected_letter]
            
            if selected_idx == correct_answers[question_num]:
                correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy, correct_count, total_count

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### 定义标准答案

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-26T05:38:44.823104Z","iopub.execute_input":"2026-02-26T05:38:44.823399Z","iopub.status.idle":"2026-02-26T05:38:44.847909Z","shell.execute_reply.started":"2026-02-26T05:38:44.823368Z","shell.execute_reply":"2026-02-26T05:38:44.847221Z"}}
# 2022年考研英语一答案 (0=A, 1=B, 2=C, 3=D)
answers_2022 = {
    1: 0,   # A. coined
    2: 2,   # C. compared
    3: 3,   # D. though
    4: 2,   # C. hinted at
    5: 3,   # D. differs
    6: 1,   # B. evidence
    7: 2,   # C. argued
    8: 1,   # B. forming
    9: 0,   # A. analogous
    10: 3,  # D. even
    11: 2,  # C. perspective
    12: 1,  # B. reducing
    13: 0,  # A. However
    14: 2,  # C. superficially
    15: 1,  # B. level
    16: 3,  # D. added
    17: 0,  # A. chances
    18: 0,  # A. danger
    19: 3,  # D. recognizes
    20: 1,  # B. poor
}

# 2023年考研英语一答案 (Caravanserais驿站文章)
answers_2023 = {
    1: 2,   # C. located
    2: 0,   # A. privately
    3: 3,   # D. combination
    4: 2,   # C. describe
    5: 2,   # C. such as
    6: 0,   # A. construction
    7: 1,   # B. faced
    8: 1,   # B. subjected
    9: 0,   # A. so that
    10: 3,  # D. meeting
    11: 3,  # D. As a result
    12: 2,  # C. exchange
    13: 2,  # C. as well as
    14: 1,  # B. influencing
    15: 0,  # A. aided
    16: 1,  # B. Indeed
    17: 3,  # D. stock up on
    18: 0,  # A. believed
    19: 3,  # D. although
    20: 0,  # A. ruins
}

# 2019年上海英语高考答案 (handwriting文章，题号1-15)
answers_2019_gaokao = {
    1: 1,   # B. dominates
    2: 0,   # A. compulsory
    3: 2,   # C. in favour of
    4: 3,   # D. majority
    5: 1,   # B. benefits
    6: 0,   # A. granting
    7: 1,   # B. driving
    8: 3,   # D. consciously
    9: 2,   # C. acquire
    10: 3,  # D. earlier
    11: 0,  # A. handwriting
    12: 2,  # C. view
    13: 3,  # D. preserving
    14: 1,  # B. counters
    15: 0,  # A. associations
}

# 2019年春季高考答案
answers_2019_spring = {
    41: 1,  # B. shorter
    42: 3,  # D. geographic
    43: 0,  # A. locals
    44: 1,  # B. accommodation
    45: 2,  # C. proposes
    46: 2,  # C. alternative
    47: 1,  # B. guidance
    48: 1,  # B. discouraging
    49: 3,  # D. ease
    50: 0,  # A. culture
    51: 2,  # C. come back
    52: 3,  # D. comparison
    53: 0,  # A. French
    54: 2,  # C. spread out
    55: 3,  # D. sustainable
}

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### 计算各测试准确率

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-26T05:38:44.848889Z","iopub.execute_input":"2026-02-26T05:38:44.849177Z","iopub.status.idle":"2026-02-26T05:38:44.868013Z","shell.execute_reply.started":"2026-02-26T05:38:44.849148Z","shell.execute_reply":"2026-02-26T05:38:44.867432Z"}}
# 计算所有4个测试的准确率
print("=" * 70)
print("BERT 完形填空准确率评估报告".center(70))
print("=" * 70)
print()

# 测试1: 2022年考研英语一
accuracy_2022, correct_2022, total_2022 = calculate_accuracy(results1, answers_2022)
print(f"📝 2022年考研英语一:  {correct_2022:2d}/{total_2022:2d} = {accuracy_2022:6.1%}")

# 测试2: 2023年考研英语一
accuracy_2023, correct_2023, total_2023 = calculate_accuracy(results2, answers_2023)
print(f"📝 2023年考研英语一:  {correct_2023:2d}/{total_2023:2d} = {accuracy_2023:6.1%}")

# 测试3: 2019年上海高考
accuracy_2019_gaokao, correct_2019_gaokao, total_2019_gaokao = calculate_accuracy(results3, answers_2019_gaokao)
print(f"📝 2019年上海高考:    {correct_2019_gaokao:2d}/{total_2019_gaokao:2d} = {accuracy_2019_gaokao:6.1%}")

# 测试4: 2019年春季高考
accuracy_2019_spring, correct_2019_spring, total_2019_spring = calculate_accuracy(results4, answers_2019_spring)
print(f"📝 2019年春季高考:    {correct_2019_spring:2d}/{total_2019_spring:2d} = {accuracy_2019_spring:6.1%}")

print()
print("-" * 70)

# 总体统计
total_correct = correct_2022 + correct_2023 + correct_2019_gaokao + correct_2019_spring
total_questions = total_2022 + total_2023 + total_2019_gaokao + total_2019_spring
total_accuracy = total_correct / total_questions if total_questions > 0 else 0

print(f"🎯 总体准确率:        {total_correct:2d}/{total_questions:2d} = {total_accuracy:6.1%}")
print("=" * 70)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 结论

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# #### 考研的由于它的上下文长度在512token之内所以不需要切掉上下文 而 高考英语需要

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### 纯底层的语言模型只是“概率的奴隶”和“语感大师”，只有跨越了从“统计高频词拼凑”到“上下文因果推理”的鸿沟（比如引入微调、树模型或思维链），AI 才能真正读懂人类的复杂逻辑。