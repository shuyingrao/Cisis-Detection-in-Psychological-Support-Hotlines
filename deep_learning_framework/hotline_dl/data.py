"""Dataset loading utilities for tabular embeddings and variable-length audio sequences."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_LABELS


@dataclass
class TabularDataset:
    """Subject-level numpy arrays for Keras/TensorFlow models."""

    subjects: np.ndarray
    X_audio: Optional[np.ndarray]
    X_text: Optional[np.ndarray]
    y: np.ndarray
    label_cols: List[str]

    @property
    def audio_dim(self) -> Optional[int]:
        return None if self.X_audio is None else int(self.X_audio.shape[1])

    @property
    def text_dim(self) -> Optional[int]:
        return None if self.X_text is None else int(self.X_text.shape[1])

    @property
    def num_labels(self) -> int:
        return int(self.y.shape[1])


@dataclass
class AudioSequenceDataset:
    """Variable-length audio sequences for PyTorch sequence models."""

    subjects: np.ndarray
    sequences: List[np.ndarray]  # each array: [seq_len, feature_dim]
    y: np.ndarray
    label_cols: List[str]

    @property
    def input_dim(self) -> int:
        if not self.sequences:
            raise ValueError("Empty sequence dataset.")
        return int(self.sequences[0].shape[1])

    @property
    def num_labels(self) -> int:
        return int(self.y.shape[1])


def audio_embedding_path(base_dir: str | Path, model_name: str, year: int | str) -> Path:
    return Path(base_dir) / "data" / "embeddings" / f"audio_embeddings_{model_name}_{year}.csv"


def text_embedding_path(base_dir: str | Path, model_name: str, year: int | str) -> Path:
    return Path(base_dir) / "data" / "embeddings" / f"text_embeddings_{model_name}_{year}.csv"


def label_path(base_dir: str | Path, year: int | str) -> Path:
    return Path(base_dir) / "data" / "labels" / f"{year}_Y.xlsx"


def _read_labels(
    path: str | Path,
    label_cols: Sequence[str] = DEFAULT_LABELS,
    sheet_name: str = "All",
    subject_col: str = "Subject",
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    missing = [c for c in [subject_col, *label_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[[subject_col, *label_cols]].drop_duplicates(subset=[subject_col])
    return df.set_index(subject_col)


def _feature_columns(df: pd.DataFrame, subject_col: str = "Subject", feature_start_col: Optional[int] = None) -> List[str]:
    """Infer embedding columns.

    If feature_start_col is provided, use df.columns[feature_start_col:]. Otherwise select numeric
    columns except the subject column. This supports files like [Subject, Segment, emb_0, ...].
    """
    if feature_start_col is not None:
        return list(df.columns[feature_start_col:])
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    return [c for c in numeric_cols if c != subject_col]


def _load_embedding_csv(
    path: str | Path,
    subject_col: str = "Subject",
    feature_start_col: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    df = pd.read_csv(path)
    if subject_col not in df.columns:
        raise ValueError(f"{path} does not contain subject column: {subject_col}")
    feat_cols = _feature_columns(df, subject_col=subject_col, feature_start_col=feature_start_col)
    if not feat_cols:
        raise ValueError(f"No numeric embedding columns found in {path}")
    return df, feat_cols


def load_tabular_dataset(
    base_dir: str | Path,
    year: int | str,
    audio_model_name: Optional[str] = None,
    text_model_name: Optional[str] = None,
    label_year: Optional[int | str] = None,
    label_cols: Sequence[str] = DEFAULT_LABELS,
    label_sheet: str = "All",
    subject_col: str = "Subject",
    audio_feature_start_col: Optional[int] = 2,
    text_feature_start_col: Optional[int] = None,
    aggregate_audio: str = "mean",
) -> TabularDataset:
    """Load subject-level text/audio embeddings and labels.

    For audio files with several segments per subject, `aggregate_audio='mean'` averages segments.
    If only one modality is required, pass None for the other modality name.
    """
    base_dir = Path(base_dir)
    label_year = year if label_year is None else label_year
    label_df = _read_labels(label_path(base_dir, label_year), label_cols, label_sheet, subject_col)

    dataframes: Dict[str, pd.DataFrame] = {}

    if audio_model_name is not None:
        df_audio, audio_cols = _load_embedding_csv(
            audio_embedding_path(base_dir, audio_model_name, year),
            subject_col=subject_col,
            feature_start_col=audio_feature_start_col,
        )
        if aggregate_audio == "mean":
            df_audio = df_audio.groupby(subject_col)[audio_cols].mean()
        elif aggregate_audio == "first":
            df_audio = df_audio.drop_duplicates(subset=[subject_col]).set_index(subject_col)[audio_cols]
        else:
            raise ValueError("aggregate_audio must be 'mean' or 'first' for tabular models.")
        dataframes["audio"] = df_audio

    if text_model_name is not None:
        df_text, text_cols = _load_embedding_csv(
            text_embedding_path(base_dir, text_model_name, year),
            subject_col=subject_col,
            feature_start_col=text_feature_start_col,
        )
        # Usually one row per subject; mean is safe if duplicated.
        df_text = df_text.groupby(subject_col)[text_cols].mean()
        dataframes["text"] = df_text

    if not dataframes:
        raise ValueError("At least one of audio_model_name or text_model_name must be provided.")

    common_subjects = label_df.index
    for df in dataframes.values():
        common_subjects = common_subjects.intersection(df.index)

    if len(common_subjects) == 0:
        raise ValueError("No common Subject found across embeddings and labels.")

    subjects = common_subjects.to_numpy()
    y = label_df.loc[common_subjects, list(label_cols)].to_numpy(dtype=np.float32)
    X_audio = None
    X_text = None
    if "audio" in dataframes:
        X_audio = dataframes["audio"].loc[common_subjects].to_numpy(dtype=np.float32)
    if "text" in dataframes:
        X_text = dataframes["text"].loc[common_subjects].to_numpy(dtype=np.float32)

    return TabularDataset(subjects=subjects, X_audio=X_audio, X_text=X_text, y=y, label_cols=list(label_cols))


def load_audio_sequence_dataset(
    base_dir: str | Path,
    year: int | str,
    audio_model_name: str,
    label_year: Optional[int | str] = None,
    label_cols: Sequence[str] = DEFAULT_LABELS,
    label_sheet: str = "All",
    subject_col: str = "Subject",
    segment_col: Optional[str] = None,
    audio_feature_start_col: Optional[int] = 2,
) -> AudioSequenceDataset:
    """Load variable-length audio segment embeddings grouped by Subject."""
    base_dir = Path(base_dir)
    label_year = year if label_year is None else label_year
    label_df = _read_labels(label_path(base_dir, label_year), label_cols, label_sheet, subject_col)
    df, feat_cols = _load_embedding_csv(
        audio_embedding_path(base_dir, audio_model_name, year),
        subject_col=subject_col,
        feature_start_col=audio_feature_start_col,
    )

    subjects: List[object] = []
    sequences: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for subject, sub_df in df.groupby(subject_col, sort=False):
        if subject not in label_df.index:
            continue
        if segment_col is not None and segment_col in sub_df.columns:
            sub_df = sub_df.sort_values(segment_col)
        seq = sub_df[feat_cols].to_numpy(dtype=np.float32)
        if seq.ndim != 2 or seq.shape[0] == 0:
            continue
        subjects.append(subject)
        sequences.append(seq)
        labels.append(label_df.loc[subject, list(label_cols)].to_numpy(dtype=np.float32))

    if not sequences:
        raise ValueError("No valid audio sequences found after Subject/label alignment.")

    return AudioSequenceDataset(
        subjects=np.asarray(subjects),
        sequences=sequences,
        y=np.vstack(labels).astype(np.float32),
        label_cols=list(label_cols),
    )


def prediction_dataframe(
    subjects: Sequence,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    label_cols: Sequence[str] = DEFAULT_LABELS,
    attention: Optional[np.ndarray] = None,
    iteration: Optional[int] = None,
) -> pd.DataFrame:
    """Build a prediction table for one iteration.

    Long-format iteration outputs are easier to aggregate later:
    each row represents one `(iteration, Subject)` prediction.
    """
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)
    data: Dict[str, Sequence] = {}
    if iteration is not None:
        data["iteration"] = [iteration] * len(subjects)
    data["Subject"] = subjects
    for i, label in enumerate(label_cols):
        data[f"prob_{label}"] = y_prob[:, i]
        data[f"pred_{label}"] = y_pred[:, i]
    if attention is not None:
        data["attention_audio_alpha"] = np.asarray(attention).reshape(-1)
    return pd.DataFrame(data)


def save_prediction_csv(
    out_path: str | Path,
    subjects: Sequence,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    label_cols: Sequence[str] = DEFAULT_LABELS,
    attention: Optional[np.ndarray] = None,
    iteration: Optional[int] = None,
) -> None:
    """Save multilabel probabilities and binary predictions for one iteration."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_dataframe(subjects, y_prob, y_pred, label_cols, attention, iteration).to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )


def save_all_iterations_prediction_csv(
    out_path: str | Path,
    prediction_frames: Sequence[pd.DataFrame],
) -> None:
    """Save predictions from all experiment iterations into one CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not prediction_frames:
        raise ValueError("prediction_frames is empty; nothing to save.")
    pd.concat(prediction_frames, ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")
