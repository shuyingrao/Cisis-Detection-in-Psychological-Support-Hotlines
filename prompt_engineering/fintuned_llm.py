import pandas as pd
import numpy as np
import json
import os
import tiktoken
import random
from openai import OpenAI
from pathlib import Path


embedding_encoding = "cl100k_base"
max_tokens = 127000  # 127000, 15700
encoding = tiktoken.get_encoding(embedding_encoding)

base_dir = ".../hotline"
sentences_path =  Path(base_dir) / "data" / "Sentences" / "2022_Y"
subjects = []
for file in os.listdir(sentences_path):
    subject = file[:-5]
    subjects.append(subject)

prompt_path = Path(base_dir) / "data" / "prompts" / "prompt_01.txt"
with open(prompt_path, 'r', encoding='utf-8') as file:
    prompt = file.read()

labels = {
    "情绪状态": ["正常", "抑郁"],
    "自杀意念": ["无", "有"],
    "自杀计划": ["无", "有"],
    "是否高危": ["非高危", "高危"]
}

label_name = ['情绪状态','自杀意念','自杀计划', '高危']
label_file = Path(base_dir) / "data" / "labels" / "2022_Y.csv"
label_data = pd.read_csv(label_file)
label_data.set_index('subject', inplace=True)
lables_df = label_data[label_name]

random.shuffle(subjects)
select_num = 50

jsonl_data_list = []
for index, subject in enumerate(subjects[:select_num]):
    test_example = pd.read_excel(sentences_path + subject + '.xlsx')
    text = str(test_example['content'].tolist())
    if len(encoding.encode(text)) > max_tokens:
        text = encoding.decode(encoding.encode(text)[:max_tokens])
    y_temp = lables_df.loc[subject]
    response_temp = {
        "情绪状态": labels["情绪状态"][y_temp["情绪状态"]],
        "自杀意念": labels["自杀意念"][y_temp["自杀意念"]],
        "自杀计划": labels["自杀计划"][y_temp["自杀计划"]],
        "是否高危": labels["是否高危"][y_temp["高危"]]
    }
    message = {
    "messages":
        [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps(response_temp, ensure_ascii=False)}
        ]
    }
    jsonl_data_list.append(json.dumps(message, ensure_ascii=False))


jsonl_train_data = "\n".join(jsonl_data_list) + "\n"

# Save to a .jsonl file
file_train_path = Path(base_dir) / "data" / "finetuned_data" / "train_data.jsonl"
with open(file_train_path, "w", encoding="utf-8") as file:
    file.write(jsonl_train_data)


client = OpenAI(api_key="sk-...")

file_train_path = Path(base_dir) / "data" / "finetuned_data" / "train_data.jsonl"

train_file = client.files.create(
  file=open(file_train_path, "rb"),
  purpose="fine-tune"
)


client.fine_tuning.jobs.create(
  training_file=train_file.id,
  model="gpt-3.5-turbo-0125",
  # hyperparameters={
  #   "n_epochs": 4,
  #   "batch_size": 16,
  #   "learning_rate_multiplier": 0.01,
  # },
)