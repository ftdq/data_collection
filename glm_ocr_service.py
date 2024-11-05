# -*- coding: utf-8 -*-
# ****************************************************
# Author:hcl
# Create: 2024/8/25
# Last Modified: 2024/8/25
# Filename: llm.py
# Description:
# ***************************************************
from zhipuai import ZhipuAI
import base64
from pdf2image import convert_from_path
import json
from tqdm import tqdm
from io import BytesIO
import os

def glm_4v(base64_string):
    try:
        response = client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "识别图片中的所有文字，不要添加任何多余的描述。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_string
                            }
                        }
                    ]
                }
            ],
            top_p=0.7,
            temperature=0.95,
            max_tokens=1024,
            stream=False
        )
        return response.choices[0].message.content
    except:
        return ""


if __name__ == '__main__':
    client = ZhipuAI(api_key="")
    # 文件路径
    file = ""
    pdf_files = sorted([os.path.join(file, f) for f in os.listdir(file) if f.endswith('.jsonl')], reverse=False)
    content = []
    for path in pdf_files:
        images = convert_from_path(path, poppler_path=r"D:\Python_practice\reptile\poppler-23.11.0\Library\bin")
        print("--------pdf加载完成--------")
        for image in tqdm(images):
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                base64_data = base64.b64encode(buffered.getvalue())
                content.append(glm_4v(base64_data.decode('utf-8')))

    with open("data/data.jsonl", "a", encoding="utf-8") as file:
        for data in content:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")
