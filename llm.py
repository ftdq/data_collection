# -*- coding: utf-8 -*-
# ****************************************************
# Author:hcl
# Create: 2024/9/1
# Last Modified: 2024/9/1
# Filename: llm.py
# Description: 
# ***************************************************
from zhipuai import ZhipuAI
from utils import process_text
import yaml
import json
from typing import List, Dict, Union


with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def glm_gen(prompt) -> str:
    client_zhipu = ZhipuAI(api_key=config["api_key"])

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client_zhipu.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            top_p=0.7,
            temperature=0.1,
            max_tokens=1024,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
    return "no answer"


def filter_travel_content(text: str, glm_gen) -> Dict[str, Union[bool, str, float]]:
    """
    使用大模型筛选与旅游相关的内容

    Args:
        text (str): 输入文本
        glm_gen: 大模型生成函数
    """
    prompt = """请判断以下文本是否与旅游相关。如果是，返回"是"；如果否，返回"否"。
    规则：
    1. 包含景点、住宿、交通、美食等旅游要素的文本视为相关
    2. 仅包含地名但无具体旅游内容的不视为相关

    需要判断的文本：
    {text}
    """
    return glm_gen(prompt.format(text=text))


def check_and_complete_text(segments: List[str], glm_gen) -> List[str]:
    """
    检查文本片段的完整性并补全缺失内容

    Args:
        segments (List[str]): 文本片段列表
        glm_gen: 大模型生成函数

    Returns:
        List[str]: 补全后的文本片段列表
    """
    completed_segments = []
    for i, segment in enumerate(segments):
        prompt = f"""请分析以下文本片段，如果发现内容不完整或存在逻辑断裂，请补充必要的信息并返回完整的文本。如果文本已经完整，直接返回原文本。

        当前文本：
        {segment}
        
        请直接返回修改后的文本或原文本："""

        response = glm_gen(prompt.format(segment=segment))
        if response != "no answer":
            completed_segments.append(segment)

    return completed_segments


def process_travel_text(text: str, glm_gen, max_len: int = 512, min_chinese_ratio: float = 0.8) -> List[str]:
    """
    完整的文本处理管道
    Args:
        text (str): 输入文本
        glm_gen: 大模型调用函数
        max_len (int): 最大文本长度

    Returns:
        List[str]: 处理后的文本片段列表
    """
    # 1. 筛选旅游相关内容
    filter_result = filter_travel_content(text, glm_gen)
    if filter_result == "no answer" or "否" in filter_result:
        return []

    # 2. 文本切分和清洗
    cleaned_segments = process_text(text, max_len, min_chinese_ratio)

    # 3. 检查完整性并补全
    completed_segments = check_and_complete_text(cleaned_segments, glm_gen)

    return list(set(completed_segments))


if __name__ == '__main__':
    json_data = []
    with open("data/data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            result = process_travel_text(text, glm_gen, config["max_len"], config["min_chinese_ratio"])
            json_data.extend(result)

    with open("data/data.txt", "w", encoding="utf-8") as f:
        for data in json_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


