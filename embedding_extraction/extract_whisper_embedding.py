import os
import torch
import torchaudio
import pandas as pd
from transformers import AutoConfig, AutoFeatureExtractor
from transformers import  WhisperModel
from pathlib import Path


base_dir = ".../hotline"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1") 

# 加载配置、特征提取器和模型
model_name = ".../whisper/openai_whisper-large-v3"
# ".../whisper/Jingmiao_whisper-small-chinese_base"
# ".../whisper/openai_whisper-small"
# ".../whisper/openai_whisper-medium"
# ".../whisper/openai_whisper-large-v3"
config = AutoConfig.from_pretrained(model_name)

extractor = AutoFeatureExtractor.from_pretrained(model_name,)
target_sampling_rate = extractor.sampling_rate

model =  WhisperModel.from_pretrained(model_name, config=config)
model = model.to(device)


def segment_audio(path, duration):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    segments = []
    for i in range(0, len(speech) - duration * target_sampling_rate, duration * target_sampling_rate):
        segments.append(speech[i:i+duration * target_sampling_rate])
    return segments

year = 2023
# 3. 音频文件路径列表（示例）
speech_list = os.listdir(Path(base_dir) / "data" / "audio" / f"{year}_Y")


# 处理每个音频文件并提取特征
results = []
dur = 10  # 每段音频的时长，单位为秒

for audio_name in speech_list:    
    audio_path = Path(base_dir) / "data" / "audio" / f"{year}_Y" / audio_name

    # dur时长分段
    for i, segment in enumerate(segment_audio(audio_path, dur)):
        input_values = extractor(segment, return_tensors="pt", sampling_rate=target_sampling_rate).input_features
        # input_values = input_values.cuda()
        input_values = input_values.to(device)

        decoder_input_ids = torch.ones([input_values.shape[0], 1], dtype=torch.long) * model.config.decoder_start_token_id
        decoder_input_ids = decoder_input_ids.to(device)
        # decoder_input_ids = decoder_input_ids.cuda()
        # 6. 将音频输入到模型中，获取音频的特征向量
        with torch.no_grad():
            outputs = model(input_values, decoder_input_ids=decoder_input_ids)

        # 7. 获取音频的隐层表示
        audio_features = outputs.last_hidden_state 

        # 将文件名和对应的音频嵌入存储到结果列表中
        embedding_vector = audio_features.squeeze().cpu().numpy()
        results.append([audio_name[:-4], i] + embedding_vector.tolist())
    

# 将结果存储到 DataFrame
feature_dim = len(embedding_vector)
columns = ["Subject", "Segment"] + [str(i+1) for i in range(feature_dim)]
df = pd.DataFrame(results, columns=columns)

# 将 DataFrame 保存到 CSV 文件
output_file = Path(base_dir) / "data" / "embeddings" / f"audio_embeddings_openai_whisper_l_{year}.csv"  # 输出文件名
df.to_csv(output_file , index=False)
print(f"音频表征向量已保存到 {output_file}")
