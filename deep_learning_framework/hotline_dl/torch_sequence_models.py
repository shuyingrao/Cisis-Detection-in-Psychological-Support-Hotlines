"""PyTorch sequence models for variable-length audio embeddings: BiLSTM and Transformer."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from .data import AudioSequenceDataset


class TorchAudioSequenceDataset(Dataset):
    def __init__(self, dataset: AudioSequenceDataset):
        self.subjects = dataset.subjects
        self.sequences = [torch.tensor(x, dtype=torch.float32) for x in dataset.sequences]
        self.labels = torch.tensor(dataset.y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx], self.subjects[idx]


def collate_audio_sequences(batch):
    sequences, labels, subjects = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long)
    labels = torch.stack(labels)
    return padded, labels, lengths, list(subjects)


class BiLSTMAudioModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_labels: int = 4, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None].to(out.device)
        mask = mask.unsqueeze(-1).float()
        avg_pool = torch.sum(out * mask, dim=1) / lengths.unsqueeze(-1).to(out.device).clamp_min(1)
        max_pool, _ = torch.max(out * mask + (1.0 - mask) * -1e9, dim=1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(combined)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerAudioModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        num_labels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_labels),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        device = x.device
        padding_mask = torch.arange(x.size(1), device=device)[None, :] >= lengths.to(device)[:, None]
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        pooled = torch.sum(out * valid_mask, dim=1) / lengths.unsqueeze(-1).to(device).clamp_min(1)
        return self.classifier(pooled)


def build_audio_sequence_model(model_type: str, input_dim: int, num_labels: int = 4) -> nn.Module:
    model_type = model_type.lower()
    if model_type in {"bilstm", "lstm"}:
        return BiLSTMAudioModel(input_dim=input_dim, num_labels=num_labels)
    if model_type in {"transformer", "tfm"}:
        return TransformerAudioModel(input_dim=input_dim, num_labels=num_labels)
    raise ValueError("model_type must be 'bilstm' or 'transformer'.")


@dataclass
class TorchTrainResult:
    best_val_f1: float
    best_state_dict: dict


def train_torch_sequence_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    patience: int = 10,
) -> TorchTrainResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by, lengths, _subjects in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = model(bx, lengths)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_prob, val_true, _ = predict_torch_sequence_model(model, val_loader, device)
        val_pred = (val_prob >= 0.5).astype(int)
        cur_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1:03d} | loss={avg_loss:.4f} | val_macro_f1={cur_f1:.4f}")

        if cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}; best val_macro_f1={best_val_f1:.4f}")
                break

    model.load_state_dict(best_state)
    return TorchTrainResult(best_val_f1=best_val_f1, best_state_dict=best_state)


def predict_torch_sequence_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List]:
    model.eval()
    probs, labels, subjects = [], [], []
    with torch.no_grad():
        for bx, by, lengths, batch_subjects in loader:
            logits = model(bx.to(device), lengths)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(by.numpy())
            subjects.extend(batch_subjects)
    return np.concatenate(probs), np.concatenate(labels), subjects
