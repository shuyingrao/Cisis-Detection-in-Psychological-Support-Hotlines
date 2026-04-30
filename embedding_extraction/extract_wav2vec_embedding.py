import os
import torch
import torchaudio
import pandas as pd
from transformers import AutoConfig, Wav2Vec2Processor
from transformers import Wav2Vec2Model
from pathlib import Path


base_dir = ".../hotline"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1") 

# 加载配置、特征提取器和模型
model_name = ".../wav2vec/jonatasgrosman_wav2vec2-large-xlsr-53-chinese-zh-cn"
config = AutoConfig.from_pretrained(model_name)

processor = Wav2Vec2Processor.from_pretrained(model_name,)
target_sampling_rate = processor.feature_extractor.sampling_rate

model = Wav2Vec2Model.from_pretrained(model_name, config=config)
model = model.to(device)

def apply_pooling(features, pooling_type='mean'):
    """
    对音频特征进行池化操作。
    :param features: 音频特征张量 (batch_size, sequence_length, hidden_size)
    :param pooling_type: 池化方式, 可以是 'mean', 'sum', 或 'max'
    :return: 池化后的音频嵌入向量
    """
    if pooling_type == 'mean':
        pooled_features = features.mean(dim=1)  # 对时间维度进行平均池化
    elif pooling_type == 'sum':
        pooled_features = features.sum(dim=1)  # 对时间维度进行求和池化
    elif pooling_type == 'max':
        pooled_features, _ = features.max(dim=1)  # 对时间维度进行最大池化
    else:
        raise ValueError("Invalid pooling type. Choose from ['mean', 'sum', 'max']")
    return pooled_features

def segment_audio(path, duration):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    segments = []
    for i in range(0, len(speech) - duration * target_sampling_rate, duration * target_sampling_rate):
        segments.append(speech[i:i+duration * target_sampling_rate])
    return segments

year = 2023
# 音频文件路径列表（示例）
speech_list = os.listdir(Path(base_dir) / "data" / "audio" / f"{year}_Y")


# 处理每个音频文件并提取特征
results = []
dur = 10 # 每段音频的时长，单位为秒
pooling_method = 'mean' 

for audio_name in speech_list:    
    audio_path = Path(base_dir) / "data" / "audio" / f"{year}_Y" / audio_name

    for i, segment in enumerate(segment_audio(audio_path, dur)):
        input_values = processor(segment, return_tensors="pt", sampling_rate=target_sampling_rate).input_values
        # input_values = input_values.cuda()
        input_values = input_values.to(device)

        # 将音频输入到模型中，获取音频的特征向量
        with torch.no_grad():
            outputs = model(input_values)

        # 获取音频的隐层表示
        audio_features = outputs.last_hidden_state 
        audio_embedding = apply_pooling(audio_features, pooling_type=pooling_method)
        # 将文件名和对应的音频嵌入存储到结果列表中
        embedding_vector = audio_embedding.squeeze().cpu().numpy()
        results.append([audio_name[:-4], i+1] + embedding_vector.tolist())


# 将结果存储到 DataFrame
feature_dim = len(embedding_vector)
columns = ["Subject", "Segment_id"] + [str(i+1) for i in range(feature_dim)]
df = pd.DataFrame(results, columns=columns)

# 将 DataFrame 保存到 CSV 文件
output_file = Path(base_dir) / "data" / "embeddings" / f"audio_embeddings_wav2vec_{year}.csv"  # 输出文件名
df.to_csv(output_file, index=False)
print(f"音频表征向量已保存到 {output_file}")