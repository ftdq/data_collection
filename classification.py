# -*- coding: utf-8 -*-
# ****************************************************
# Author:hcl
# Create: 2024/9/9
# Last Modified: 2024/9/9
# Filename: classification.py
# Description: 
# ***************************************************
import json

import torch
from sentence_transformers import SentenceTransformer
import numpy as np


def main():
    model = SentenceTransformer("embedding_model/MiniCPM-Embedding", trust_remote_code=True,
                                model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16})
    model.max_seq_length = 512
    model.tokenizer.padding_side = "right"

    with open("/data/data.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()

    INSTRUCTION = "Query: "
    queries = ["判断文本内容是否和广西文旅相关，包括但不限于美食、景点、攻略、住宿、购物等。"]
    embeddings_query = model.encode(queries, prompt=INSTRUCTION, normalize_embeddings=True)

    batch_size = 10
    num_chunks = (len(texts) + batch_size - 1) // batch_size
    for i in range(num_chunks):
        start_index = i * batch_size
        end_index = start_index + batch_size
        chunk = texts[start_index:end_index]
        embeddings_doc = model.encode(chunk, normalize_embeddings=True)
        scores = (embeddings_query @ embeddings_doc.T)
        indices = np.argwhere(scores > 0.8).tolist()[0]
        result = [chunk[idx] for idx in indices]

    with open("output.txt", "a", encoding="utf-8") as file:
        for data in result:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()