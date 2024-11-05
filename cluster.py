# -*- coding: utf-8 -*-
# ****************************************************
# Author:hcl
# Create: 2024/9/5
# Last Modified: 2024/9/5
# Filename: cluster.py
# Description: 
# ***************************************************
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict
import json


def cluster_texts(embeddings: np.ndarray, texts: List[str], n_clusters: int = 5) -> List[Dict[int, List[str]]]:
    """
    对文本进行聚类并返回每个簇的文本列表

    Args:
        embeddings: 文本向量矩阵
        texts: 原始文本列表
        n_clusters: 聚类数量

    Returns:
        Dict[int, List[str]]: 簇ID到文本列表的映射
    """
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 整理聚类结果
    result_list = []
    for i in range(n_clusters):
        cluster_dict = {}
        mask = cluster_labels == i
        cluster_dict[i] = np.array(texts)[mask].tolist()
        result_list.append(cluster_dict)

    return result_list


def main():
    model = SentenceTransformer("/embedding_model/MiniCPM-Embedding", trust_remote_code=True,
                                model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16})
    model.max_seq_length = 512
    model.tokenizer.padding_side = "right"

    # embedding
    with open("data/data.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()
    embeddings = model.encode(texts, normalize_embeddings=True)

    # 聚类
    results = cluster_texts(embeddings, texts)
    with open("data/results.jsonl", "w", encoding="utf-8") as f:
        for data in results:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
