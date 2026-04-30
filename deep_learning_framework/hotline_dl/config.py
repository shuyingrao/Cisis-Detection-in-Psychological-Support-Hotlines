"""Central configuration for hotline multilabel experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DEFAULT_LABELS = ["情绪状态", "自杀意念", "自杀计划", "高危"]

# 常用 embedding 维度。实际运行时也会从 CSV 自动推断维度。
TEXT_MODEL_DIMS = {
    "roberta": 768,      # RoBERTa [CLS]
    "gpt": 3072,         # GPT embedding
}

AUDIO_MODEL_DIMS = {
    "hubert": 768,
    "whisper": 768,
    "openai_whisper": 1024,
    "openai_whisper_s": 768,
    "openai_whisper_m": 1024,
    "openai_whisper_l": 1280,
    "wav2vec": 1024,
}


@dataclass
class ExperimentConfig:
    """Reusable experiment options.

    Expected project layout by default:
        base_dir/
          data/
            embeddings/
              audio_embeddings_{audio_model}_{year}.csv
              text_embeddings_{text_model}_{year}.csv
            labels/
              {year}_Y.xlsx
          outputs/
    """

    base_dir: str = "."
    train_year: int = 2022
    test_year: int = 2023
    audio_model: str = "openai_whisper_s"
    text_model: str = "gpt"
    labels: List[str] = field(default_factory=lambda: DEFAULT_LABELS.copy())
    label_sheet: str = "All"
    subject_col: str = "Subject"
    segment_col: Optional[str] = None
    output_dir: str = "outputs"
    seed: int = 42

    @property
    def base_path(self) -> Path:
        return Path(self.base_dir).expanduser().resolve()

    @property
    def output_path(self) -> Path:
        p = self.base_path / self.output_dir
        p.mkdir(parents=True, exist_ok=True)
        return p
