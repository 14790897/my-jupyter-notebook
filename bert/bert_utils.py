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
