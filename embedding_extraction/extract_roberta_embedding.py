from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
import pandas as pd
import numpy as np
import json
from pathlib import Path

year = 2023
base_dir = ".../hotline"
transcript_json_dir = Path(base_dir) / "data" / "transcript" / f"transcript_{year}_Y.json"
subject_list = []
text_data = []
f = open(transcript_json_dir, 'r', encoding='utf-8')
for line in f.readlines():
    dic = json.loads(line)
    subject_list.append(dic['subject'])
    text_data.append(dic['doc_text'])

bert_model = ".../BERT/chinese-roberta-wwm-ext"
# # 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = TFBertModel.from_pretrained(bert_model)

embeddings = []
for text_i in text_data:
    # 编码文本
    encoded_input = tokenizer(text_i, max_length=10000, return_tensors='tf')
    # 获取输出结果
    output = model(encoded_input)
    # 获取最后一层的隐藏状态，这是模型的输出
    embeddings = output.last_hidden_state.numpy()
    # mean, max
    # embeddings.append(np.max(embeddings[0,1:-1,:],axis=0))
    # CLS token    
    embeddings.append(embeddings[0,0,:])

df_data = pd.DataFrame(embeddings, index=subject_list)
out_dir = Path(base_dir) / "data" / "embeddings" / f"text_embeddings_roberta_{year}.csv"
df_data.to_csv(out_dir, index=True)
