# -*- coding: utf-8 -*-
# ****************************************************
# Author:hcl
# Create: 2024/8/26
# Last Modified: 2024/8/26
# Filename: utils.py
# Description:
# ***************************************************
import re
from typing import List


def process_text(text: str, max_len: int = 512, min_chinese_ratio: float = 0.8) -> List[str]:
    """
    处理长文本：清洗、分句、验证中文比例

    Args:
        text (str): 输入文本
        max_len (int): 每个子句的最大长度
        min_chinese_ratio (float): 最小中文字符占比

    Returns:
        List[str]: 处理后的文本片段列表

    Raises:
        ValueError: 当输入文本为空或参数无效时
    """
    if not text or not isinstance(text, str):
        raise ValueError("输入文本不能为空且必须是字符串类型")
    if max_len <= 0:
        raise ValueError("max_len必须大于0")
    if not 0 <= min_chinese_ratio <= 1:
        raise ValueError("min_chinese_ratio必须在0到1之间")

    text = _clean_text(text)
    sentences = _split_sentences(text)
    result = _merge_and_validate_sentences(sentences, max_len, min_chinese_ratio)

    return result


def _clean_text(text: str) -> str:
    cleanup_patterns = {
        'spaces': (r'[   　‎\u200F\u200D\u2006\uFEFF\u2029\u2028\u00A0\u000D\u000C\u000B\u0009\u0008]', ''),
        'punctuation': (r'[()（）{}【】_.□￥\n\s\"]', ''),
        'consecutive_puncts': (r'([^\w\s])([^\w\s])', r'\2'),
        'urls': (r'https?://\S+', ''),
        'phone_numbers': (r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|\d{3}[-.\s]?\d{4}', ''),
        'emails': (r'\S+@\S+\.\S+', ''),
        'multiple_spaces': (r'\s+', ' ')
    }

    for pattern, replacement in cleanup_patterns.items():
        text = re.sub(pattern[0], pattern[1], text)

    return text.strip()


def _split_sentences(text: str) -> List[str]:
    pattern = f'[^。！？!?]+[。！？!?]+|[^。！？!?]+$'

    sentences = re.findall(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _check_chinese_ratio(text: str, min_ratio: float) -> bool:
    """检查中文字符占比"""
    if not text:
        return False

    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    return (chinese_chars / len(text)) >= min_ratio


def _merge_and_validate_sentences(sentences: List[str], max_len: int, min_chinese_ratio: float) -> List[str]:
    """合并句子并验证中文比例"""
    result = []
    current_text = ""

    for sentence in sentences:
        potential_text = current_text + sentence if current_text else sentence

        if len(potential_text) <= max_len:
            current_text = potential_text
        else:
            if current_text and _check_chinese_ratio(current_text, min_chinese_ratio):
                result.append(current_text)
            current_text = sentence

    # 处理最后一个片段
    if current_text and _check_chinese_ratio(current_text, min_chinese_ratio):
        result.append(current_text)

    return result

