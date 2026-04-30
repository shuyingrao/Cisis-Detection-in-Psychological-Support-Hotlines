from openai import OpenAI
import pandas as pd
import numpy as np
import tiktoken
import time
from pathlib import Path


client = OpenAI(api_key="sk-...")


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# embedding model parameters
year = 2023
embedding_model = "text-embedding-3-large"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
encoding = tiktoken.get_encoding(embedding_encoding)

base_dir = ".../hotline"
transcript_dir = Path(base_dir) / "data" / "transcript" / f"transcript_{year}_Y.csv"
data = pd.read_csv(transcript_dir, encoding='gb18030')

data['len'] = data.Content.apply(lambda x: len(x))
data["n_tokens"] = data.Content.apply(lambda x: len(encoding.encode(x)))
data = data.set_index('Subject')

shorten_data = pd.DataFrame()
shorten_data['text'] = data.Content.apply(lambda x: encoding.decode(encoding.encode(x)[:max_tokens]) if len(encoding.encode(x)) > max_tokens else x)
shorten_data["n_tokens"] = shorten_data.text.apply(lambda x: len(encoding.encode(x)))
# shorten_data['ada_embedding'] = shorten_data.text.apply(lambda x: get_embedding(x, model=embedding_model))

s = []
for text in shorten_data.text:
    embedding = get_embedding(text, model=embedding_model)
    s.append(embedding)

df = pd.DataFrame(np.array(s), index=shorten_data.index)
out_dir = Path(base_dir) / "data" / "embeddings" / f"text_embeddings_gpt_{year}.csv"
df.to_csv(out_dir, index=True)
