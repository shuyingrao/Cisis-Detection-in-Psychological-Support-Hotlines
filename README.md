# Cisis-Detection-in-Psychological-Support-Hotlines

This repository contains the implementation code for the paper titled **"Automating Multi-label Crisis Detection in Psychological Support Hotlines with Pre-trained Models"**. The project supports a full experimental workflow from raw hotline audio preparation to embedding extraction, deep-learning prediction, and LLM-based prompt engineering.

The target task is **multi-label crisis detection** for psychological support hotline calls. Each call is assigned four binary labels:

- `情绪状态` — emotional state
- `自杀意念` — suicidal ideation
- `自杀计划` — suicide plan
- `高危` — high risk

The default experimental setting trains models on **2022** data and generates multi-label predictions for **2023** data.


## Citation

If you use this repositoryor or find this repository useful for your research or work, please cite us using the following citation:

```text
@article{shoham2024cpllm,
  title={Automating Multi-label Crisis Detection in Psychological Support Hotlines with Pre-trained Models},
  author={Rao S, Deng G, Song H, Chen Q, Luo M, Zhang Y, et al.},
  journal={PLOS Digital Health},
  volume={5},
  number={4},
  pages={e0001383},
  year={2026}
}
```

---

## Main Workflow

The code is organized around four major stages:

1. **Data preparation**: `prepare_data/`
2. **Embedding extraction**: `embedding_extraction/`
3. **Deep-learning framework based on pre-trained embeddings**: `deep_learning_framework/`
4. **LLM prompt engineering and explanation generation**: `prompt_engineering/`

---

## Getting Started

### 1. Clone or Unpack the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

If you received the code as a compressed archive, unpack it and enter the project root directory.

---

### 2. Create the Environment

An `environment.yml` file will be added later. Once it is available, create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate crisis_detection
```

### 3. Configure Local Paths and Credentials

Several scripts contain project-specific placeholders. Before running the pipeline, update the following values in the relevant scripts:

```python
base_dir = ".../hotline"
```

Replace it with the absolute path to the project root, for example:

```python
base_dir = "/path/to/crisis_detection_in_hotlines"
```

You should also configure local or remote model paths, GPU devices, and API credentials where needed:

- Hugging Face / local RoBERTa model path in `embedding_extraction/extract_roberta_embedding.py`
- wav2vec model path in `embedding_extraction/extract_wav2vec_embedding.py`
- Whisper model path in `embedding_extraction/extract_whisper_embedding.py`
- Pyannote VAD model path in `prepare_data/preprocess_audio.py`
- Speech-to-text API credentials in `prepare_data/audio_to_transcription.py`
- OpenAI API key in `embedding_extraction/extract_gpt_embedding.py` and `prompt_engineering/*.py`

Do **not** commit private API keys or credentials to version control.

---

## Stage 1: Data Preparation (`prepare_data/`)

### 1.1 Audio Preprocessing

`prepare_data/preprocess_audio.py` performs the following operations:

1. Converts raw MP3 audio to mono WAV.
2. Applies voice activity detection using Pyannote.
3. Concatenates detected speech segments.
4. Applies noise reduction.
5. Performs loudness normalization.
6. Saves processed WAV files.

Run:

```bash
python prepare_data/preprocess_audio.py
```

Default input and output paths:

```text
Input:  data/raw_audio/{year}_Y/
Middle: data/mono_audio/{year}_Y/
Output: data/audio/{year}_Y/
```

Before running, update `base_dir`, the year-specific folders, GPU device, and the Pyannote model path.

---

### 1.2 Audio Transcription

`prepare_data/audio_to_transcription.py` converts processed audio into sentence-level transcripts and full-call transcript files.

The script performs the following steps:

1. Uploads processed WAV files to the speech-to-text service.
2. Receives sentence-level transcription results.
3. Saves each call transcript as an Excel file.
4. Creates sentence-level files under `data/Sentences/{year}_Y/`.
5. Creates full-call transcript files:

```text
data/transcript/transcript_{year}_Y.csv
data/transcript/transcript_{year}_Y.json
```

Run:

```bash
python prepare_data/audio_to_transcription.py
```

Before running, update:

- `base_dir`
- `appid`
- `secret_key`
- target year and input/output folders

---

## Stage 2: Embedding Extraction (`embedding_extraction/`)

This stage converts transcript text and processed audio into fixed-dimensional embeddings for downstream models.

### 2.1 GPT Text Embeddings

`embedding_extraction/extract_gpt_embedding.py` extracts OpenAI GPT text embeddings from full-call transcripts.

Run:

```bash
python embedding_extraction/extract_gpt_embedding.py
```

Default output:

```text
data/embeddings/text_embeddings_gpt_{year}.csv
```

Before running, configure:

- `base_dir`
- `year`
- `OpenAI(api_key=...)`
- `embedding_model`, for example `text-embedding-3-large`

---

### 2.2 RoBERTa `[CLS]` Text Embeddings

`embedding_extraction/extract_roberta_embedding.py` extracts RoBERTa `[CLS]` embeddings from JSON transcripts.

Run:

```bash
python embedding_extraction/extract_roberta_embedding.py
```

Default output:

```text
data/embeddings/text_embeddings_roberta_{year}.csv
```

Before running, configure:

- `base_dir`
- `year`
- local or Hugging Face RoBERTa model path

---

### 2.3 wav2vec Audio Embeddings

`embedding_extraction/extract_wav2vec_embedding.py` extracts segment-level wav2vec audio embeddings.

Run:

```bash
python embedding_extraction/extract_wav2vec_embedding.py
```

Default output:

```text
data/embeddings/audio_embeddings_wav2vec_{year}.csv
```

The output format is:

```text
Subject, Segment_id, feature_1, feature_2, ..., feature_D
```

Before running, configure:

- `base_dir`
- `year`
- wav2vec model path
- GPU device
- segment duration `dur`
- pooling method

---

### 2.4 Whisper Audio Embeddings

`embedding_extraction/extract_whisper_embedding.py` extracts segment-level Whisper audio embeddings.

Run:

```bash
python embedding_extraction/extract_whisper_embedding.py
```

Default output:

```text
data/embeddings/audio_embeddings_openai_whisper_l_{year}.csv
```

The output format is:

```text
Subject, Segment, feature_1, feature_2, ..., feature_D
```

Before running, configure:

- `base_dir`
- `year`
- Whisper model path
- GPU device
- segment duration `dur`

---

## Stage 3: Deep-Learning Framework (`deep_learning_framework/`)

The deep-learning framework trains models on 2022 data and generates multi-label predictions for 2023 data.

Supported model families:

1. Text-only CNN models
   - GPT embeddings
   - RoBERTa `[CLS]` embeddings
2. Audio-only CNN models
   - wav2vec embeddings
   - Whisper embeddings
3. Audio sequence models
   - BiLSTM
   - Transformer
4. Text-audio fusion model
   - Attention-based fusion mechanism

The framework saves predictions from all independent iterations. It does not compute or save test-performance metrics in the current version.

### 3.1 TensorFlow / Keras Models

Use `deep_learning_framework/train_tf.py` for:

- `text_cnn`
- `audio_cnn`
- `fusion_attention`

#### GPT Text CNN

```bash
cd deep_learning_framework
python train_tf.py \
  --base-dir .. \
  --model-type text_cnn \
  --text-model gpt \
  --train-year 2022 \
  --test-year 2023 \
  --num-runs 100
```

#### RoBERTa Text CNN

```bash
cd deep_learning_framework
python train_tf.py \
  --base-dir .. \
  --model-type text_cnn \
  --text-model roberta \
  --train-year 2022 \
  --test-year 2023 \
  --num-runs 100
```

#### wav2vec Audio CNN

```bash
cd deep_learning_framework
python train_tf.py \
  --base-dir .. \
  --model-type audio_cnn \
  --audio-model wav2vec \
  --train-year 2022 \
  --test-year 2023 \
  --num-runs 100
```

#### Whisper Audio CNN

```bash
cd deep_learning_framework
python train_tf.py \
  --base-dir .. \
  --model-type audio_cnn \
  --audio-model openai_whisper_l \
  --train-year 2022 \
  --test-year 2023 \
  --num-runs 100
```

#### Text + Audio Attention Fusion

```bash
cd deep_learning_framework
python train_tf.py \
  --base-dir .. \
  --model-type fusion_attention \
  --text-model gpt \
  --audio-model openai_whisper_l \
  --train-year 2022 \
  --test-year 2023 \
  --num-runs 100
```

Main options:

```text
--model-type      text_cnn | audio_cnn | fusion_attention
--text-model      gpt | roberta
--audio-model     wav2vec | openai_whisper_l | openai_whisper_s | openai_whisper_m | whisper
--epochs          number of training epochs; default: 200
--batch-size      batch size; default: 32
--lr              learning rate; default: 1e-3
--patience        early-stopping patience; default: 40
--threshold       prediction threshold; default: 0.5
--val-size        validation split ratio; default: 0.1
--seed            base random seed; default: 42
--num-runs        number of independent training/prediction iterations; default: 100
--save-models     optionally save one model per iteration
```

Prediction output:

```text
outputs/{model_type}/{used_modalities}/all_iterations_test_{test_year}_multilabel_predictions.csv
```

Examples:

```text
outputs/text_cnn/text-gpt/all_iterations_test_2023_multilabel_predictions.csv
outputs/text_cnn/text-roberta/all_iterations_test_2023_multilabel_predictions.csv
outputs/audio_cnn/audio-wav2vec/all_iterations_test_2023_multilabel_predictions.csv
outputs/fusion_attention/text-gpt_audio-openai_whisper_l/all_iterations_test_2023_multilabel_predictions.csv
```

Each row corresponds to one subject in one iteration:

```text
iteration, Subject,
prob_情绪状态, pred_情绪状态,
prob_自杀意念, pred_自杀意念,
prob_自杀计划, pred_自杀计划,
prob_高危, pred_高危
```

For the fusion model, the output also includes:

```text
attention_audio_alpha
```

---

### 3.2 PyTorch Audio Sequence Models

Use `deep_learning_framework/train_torch_sequence.py` for segment-level audio sequence modeling.

Supported model types:

- `bilstm`
- `transformer`

#### BiLSTM

```bash
cd deep_learning_framework
python train_torch_sequence.py \
  --base-dir .. \
  --model-type bilstm \
  --audio-model wav2vec \
  --train-year 2022 \
  --test-year 2023 \
  --num-runs 100
```

#### Transformer

```bash
cd deep_learning_framework
python train_torch_sequence.py \
  --base-dir .. \
  --model-type transformer \
  --audio-model openai_whisper_l \
  --train-year 2022 \
  --test-year 2023 \
  --num-runs 100
```

Main options:

```text
--model-type      bilstm | transformer
--audio-model     wav2vec | openai_whisper_l | openai_whisper_s | openai_whisper_m | whisper
--epochs          number of training epochs; default: 200
--batch-size      batch size; default: 64
--lr              learning rate; default: 1e-4
--patience        early-stopping patience; default: 10
--threshold       prediction threshold; default: 0.5
--val-size        validation split ratio; default: 0.2
--seed            base random seed; default: 42
--num-runs        number of independent training/prediction iterations; default: 100
--segment-col     optional segment column for sorting within each subject
--save-models     optionally save one model per iteration
```

Prediction output:

```text
outputs/audio_sequence_{model_type}/{audio_model}/all_iterations_test_{test_year}_multilabel_predictions.csv
```

Examples:

```text
outputs/audio_sequence_bilstm/wav2vec/all_iterations_test_2023_multilabel_predictions.csv
outputs/audio_sequence_transformer/openai_whisper_l/all_iterations_test_2023_multilabel_predictions.csv
```

Column format:

```text
iteration
Subject
prob_情绪状态
pred_情绪状态
prob_自杀意念
pred_自杀意念
prob_自杀计划
pred_自杀计划
prob_高危
pred_高危
```

The `prob_*` columns contain sigmoid probabilities. The `pred_*` columns contain binary predictions generated using the selected threshold.

---

## Stage 4: Prompt Engineering and LLM-based Prediction (`prompt_engineering/`)

The `prompt_engineering/` directory provides scripts for LLM-based crisis detection and explanation generation.

### 4.1 Few-shot or Fine-tuned LLM Inference

`prompt_engineering/few_shot_learning.py` calls an LLM with the prompt template in:

```text
data/prompts/prompt_01.txt
```

It reads sentence-level transcripts from:

```text
data/Sentences/{year}_Y/
```

and saves responses to:

```text
data/gpt_responses_{year}_Y/
```

Run:

```bash
python prompt_engineering/few_shot_learning.py
```

Before running, configure:

- `base_dir`
- `year`
- OpenAI API key
- target LLM or fine-tuned model name
- number of repeated runs

---

### 4.2 Fine-tuning Data Construction and Job Submission

`prompt_engineering/fintuned_llm.py` constructs a JSONL training file from 2022 labeled data and submits a fine-tuning job.

Default JSONL output:

```text
data/finetuned_data/train_data.jsonl
```

Run:

```bash
python prompt_engineering/fintuned_llm.py
```

Before running, configure:

- `base_dir`
- OpenAI API key
- training labels under `data/labels/2022_Y.csv`
- prompt file under `data/prompts/prompt_01.txt`
- fine-tuning base model
- number of selected examples

---

### 4.3 Explanation Generation

`prompt_engineering/generation_explanation.py` generates explanation prompts from model predictions and transcript content.

The script performs majority voting across repeated predictions and formats the predicted labels with the original call text for explanation generation.

Run:

```bash
python prompt_engineering/generation_explanation.py
```

Before running, configure:

- prediction file path
- transcript sentence directory
- label file path
- output file path

---

## Expected Data Layout

The code expects the following data layout under `base_dir`:

```text
data/
├── raw_audio/
│   ├── 2022_Y/
│   └── 2023_Y/
├── mono_audio/
│   ├── 2022_Y/
│   └── 2023_Y/
├── audio/
│   ├── 2022_Y/
│   └── 2023_Y/
├── transcript/
│   ├── 2022_Y/
│   ├── 2023_Y/
│   ├── transcript_2022_Y.csv
│   ├── transcript_2022_Y.json
│   ├── transcript_2023_Y.csv
│   └── transcript_2023_Y.json
├── Sentences/
│   ├── 2022_Y/
│   └── 2023_Y/
├── embeddings/
│   ├── text_embeddings_gpt_2022.csv
│   ├── text_embeddings_gpt_2023.csv
│   ├── text_embeddings_roberta_2022.csv
│   ├── text_embeddings_roberta_2023.csv
│   ├── audio_embeddings_wav2vec_2022.csv
│   ├── audio_embeddings_wav2vec_2023.csv
│   ├── audio_embeddings_openai_whisper_l_2022.csv
│   └── audio_embeddings_openai_whisper_l_2023.csv
├── labels/
│   ├── 2022_Y.xlsx
│   ├── 2023_Y.xlsx
│   ├── 2022_Y.csv
│   └── 2023_Y.csv
├── prompts/
│   └── prompt_01.txt
├── finetuned_data/
└── gpt_responses_2023_Y/
```

The deep-learning framework expects label files in Excel format by default:

```text
data/labels/2022_Y.xlsx
data/labels/2023_Y.xlsx
```

The label sheet is expected to contain a `Subject` column and the four label columns:

```text
Subject, 情绪状态, 自杀意念, 自杀计划, 高危
```

---

## Reproducing the Default Experiment

A typical end-to-end workflow is:

```bash
# 1. Preprocess raw audio
python prepare_data/preprocess_audio.py

# 2. Transcribe processed audio
python prepare_data/audio_to_transcription.py

# 3. Extract text embeddings
python embedding_extraction/extract_gpt_embedding.py
python embedding_extraction/extract_roberta_embedding.py

# 4. Extract audio embeddings
python embedding_extraction/extract_wav2vec_embedding.py
python embedding_extraction/extract_whisper_embedding.py

# 5. Train deep-learning models and save all 2023 predictions
cd deep_learning_framework
python train_tf.py --base-dir .. --model-type text_cnn --text-model gpt --num-runs 100
python train_tf.py --base-dir .. --model-type text_cnn --text-model roberta --num-runs 100
python train_tf.py --base-dir .. --model-type audio_cnn --audio-model wav2vec --num-runs 100
python train_tf.py --base-dir .. --model-type fusion_attention --text-model gpt --audio-model openai_whisper_l --num-runs 100
python train_torch_sequence.py --base-dir .. --model-type bilstm --audio-model wav2vec --num-runs 100
python train_torch_sequence.py --base-dir .. --model-type transformer --audio-model openai_whisper_l --num-runs 100
cd ..

# 6. Run LLM-based prompt engineering experiments
python prompt_engineering/few_shot_learning.py
python prompt_engineering/generation_explanation.py
```

## Security and Privacy

This repository is designed for sensitive psychological hotline data. Follow the privacy and security requirements of your institution or ethics approval. In particular:

- Do not publish raw audio, transcripts, labels, or personally identifiable information.
- Do not commit API keys, model credentials, or private server paths.
- Store raw and processed data outside public repositories.
- Review all generated outputs before sharing.

---

