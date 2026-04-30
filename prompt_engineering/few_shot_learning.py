from openai import OpenAI
import pandas as pd
import numpy as np
import tiktoken
import time
import re
import json
import os
from pathlib import Path


def get_set_name(json_path):
    s = []
    for file in json_path:
        f = open(file, 'r', encoding='utf-8')
        for line in f.readlines():
            dic = json.loads(line)
            s.append(dic['subject'])
    return s

year = 2023
base_dir = ".../hotline"
sentences_path =  Path(base_dir) / "data" / "Sentences" / f"{year}_Y"

prompt_path =  Path(base_dir) / "data" / "prompts" / "prompt_01.txt"
with open(prompt_path, 'r', encoding='utf-8') as file:
    prompt = file.read()


client = OpenAI(api_key="sk-...")
def get_response(text, model= "ft:gpt-3.5-turbo-0125:personal::9o7GjI4P"):
    # gpt-4-turbo gpt-4o-2024-05-13
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text
            }
        ], )
    return response.choices[0].message.content


embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)
for i in range(4):
    print(f"第{i+1}次运行\n")
    start_time = time.time()
    s = []
    for file in os.listdir(sentences_path):
        subject = file[:-5]
        test_example = pd.read_excel(sentences_path + file)
        text = str(test_example['content'].tolist())
        max_tokens = 16000 - len(encoding.encode(prompt))
        if len(encoding.encode(text)) > max_tokens:
            text = encoding.decode(encoding.encode(text)[:max_tokens])
        response = get_response(text)

        dict1 = {}
        dict1['subject'] = subject
        dict1['response'] = response
        # dict1['response'] = response[7:-3]
        s.append(json.dumps(dict1, ensure_ascii=False))

    end_time = time.time()
    # 计算总运行时间
    total_time = end_time - start_time
    # 输出总运行时间
    print(f"代码的总运行时间为：{total_time}秒\n")

    # %%
    output_path1 =  Path(base_dir) / "data" / f"gpt_responses_{year}_Y" / f"finetuned_gpt_3_5_{i+1}.json"
    with open(output_path1, "w+", encoding='utf-8') as f:
        for line in s:
            f.write('%s\n' % line)
    print(f"saved results to {output_path1}\n")
