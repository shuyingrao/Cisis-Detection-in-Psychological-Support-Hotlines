"""Train PyTorch BiLSTM/Transformer audio sequence models repeatedly and save all 2023 predictions.

This script no longer computes or saves test performance metrics.  It trains the selected
model for `--num-runs` independent iterations and stores every iteration's multilabel
prediction on the 2023 test set in one long-format CSV.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from hotline_dl.data import (
    load_audio_sequence_dataset,
    prediction_dataframe,
    save_all_iterations_prediction_csv,
)
from hotline_dl.metrics import binarize
from hotline_dl.torch_sequence_models import (
    TorchAudioSequenceDataset,
    build_audio_sequence_model,
    collate_audio_sequences,
    predict_torch_sequence_model,
    train_torch_sequence_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train audio sequence model repeatedly: BiLSTM or Transformer.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--model-type", choices=["bilstm", "transformer"], required=True)
    p.add_argument("--audio-model", default="openai_whisper_s")
    p.add_argument("--train-year", type=int, default=2022)
    p.add_argument("--test-year", type=int, default=2023)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-runs", type=int, default=100, help="Number of independent train/predict iterations.")
    p.add_argument("--segment-col", default=None, help="Optional segment/time column for sorting within Subject.")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument(
        "--save-models",
        action="store_true",
        help="Optionally save one model file per iteration. Disabled by default to keep outputs small.",
    )
    return p.parse_args()


def reset_random_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_iteration_loaders(torch_train, torch_test, args, iteration: int):
    iter_seed = args.seed + iteration - 1
    train_idx, val_idx = train_test_split(
        range(len(torch_train)),
        test_size=args.val_size,
        random_state=iter_seed,
        shuffle=True,
    )
    generator = torch.Generator()
    generator.manual_seed(iter_seed)
    train_loader = DataLoader(
        Subset(torch_train, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_audio_sequences,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(torch_train, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_audio_sequences,
    )
    test_loader = DataLoader(
        torch_test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_audio_sequences,
    )
    return train_loader, val_loader, test_loader


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = base_dir / args.output_dir / f"audio_sequence_{args.model_type}" / args.audio_model
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_audio_sequence_dataset(base_dir, args.train_year, args.audio_model, segment_col=args.segment_col)
    test_data = load_audio_sequence_dataset(base_dir, args.test_year, args.audio_model, segment_col=args.segment_col)

    torch_train = TorchAudioSequenceDataset(train_data)
    torch_test = TorchAudioSequenceDataset(test_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_frames = []
    for iteration in range(1, args.num_runs + 1):
        print(f"\n===== Iteration {iteration}/{args.num_runs} =====")
        iter_seed = args.seed + iteration - 1
        reset_random_state(iter_seed)
        train_loader, val_loader, test_loader = build_iteration_loaders(torch_train, torch_test, args, iteration)

        model = build_audio_sequence_model(args.model_type, input_dim=train_data.input_dim, num_labels=train_data.num_labels).to(device)
        result = train_torch_sequence_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
        )

        if args.save_models:
            model_path = out_dir / f"model_run_{iteration:03d}.pt"
            torch.save(result.best_state_dict, model_path)

        y_prob, _y_true, subjects = predict_torch_sequence_model(model, test_loader, device)
        y_pred = binarize(y_prob, args.threshold)
        prediction_frames.append(
            prediction_dataframe(
                subjects=subjects,
                y_prob=y_prob,
                y_pred=y_pred,
                label_cols=train_data.label_cols,
                iteration=iteration,
            )
        )

    prediction_path = out_dir / f"all_iterations_test_{args.test_year}_multilabel_predictions.csv"
    save_all_iterations_prediction_csv(prediction_path, prediction_frames)
    print(f"\nSaved all iteration predictions to: {prediction_path}")


if __name__ == "__main__":
    main()
