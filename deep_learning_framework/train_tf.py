"""Train TensorFlow/Keras CNN models repeatedly and save all 2023 predictions.

This script no longer computes or saves test performance metrics.  It trains the selected
model for `--num-runs` independent iterations and stores every iteration's multilabel
prediction on the 2023 test set in one long-format CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from hotline_dl.data import (
    load_tabular_dataset,
    prediction_dataframe,
    save_all_iterations_prediction_csv,
)
from hotline_dl.metrics import binarize
from hotline_dl.tf_models import AttentionFusionModel, UnimodalCNNModel


def parse_args():
    p = argparse.ArgumentParser(description="Train subject-level CNN or attention-fusion models repeatedly.")
    p.add_argument("--base-dir", default=".", help="Project root containing data/embeddings and data/labels.")
    p.add_argument("--model-type", choices=["text_cnn", "audio_cnn", "fusion_attention"], required=True)
    p.add_argument("--text-model", default="gpt", help="gpt or roberta, used by text/fusion models.")
    p.add_argument("--audio-model", default="openai_whisper_s", help="wav2vec / whisper / openai_whisper_s/m/l etc.")
    p.add_argument("--train-year", type=int, default=2022)
    p.add_argument("--test-year", type=int, default=2023)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-runs", type=int, default=100, help="Number of independent train/predict iterations.")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument(
        "--save-models",
        action="store_true",
        help="Optionally save one model file per iteration. Disabled by default to keep outputs small.",
    )
    return p.parse_args()


def build_run_dir_name(model_type: str, text_model: str, audio_model: str) -> str:
    """Return a run directory name containing only actually used modalities."""
    if model_type == "text_cnn":
        return f"text-{text_model}"
    if model_type == "audio_cnn":
        return f"audio-{audio_model}"
    if model_type == "fusion_attention":
        return f"text-{text_model}_audio-{audio_model}"
    raise ValueError(f"Unsupported model_type: {model_type}")


def reset_random_state(seed: int) -> None:
    """Reset TensorFlow/Numpy random states before each iteration."""
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()


def train_predict_one_iteration(args, train_ds, test_ds, iteration: int):
    """Train one model instance and return 2023 probabilities plus optional attention."""
    iter_seed = args.seed + iteration - 1
    reset_random_state(iter_seed)

    if args.model_type == "text_cnn":
        X_train, X_test = train_ds.X_text, test_ds.X_text
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            train_ds.y,
            test_size=args.val_size,
            random_state=iter_seed,
            shuffle=True,
        )
        model = UnimodalCNNModel(input_dim=train_ds.text_dim, modality="text", learning_rate=args.lr)
        model.train(X_tr, y_tr, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
        y_prob = model.predict(X_test)
        attention: Optional[np.ndarray] = None

    elif args.model_type == "audio_cnn":
        X_train, X_test = train_ds.X_audio, test_ds.X_audio
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            train_ds.y,
            test_size=args.val_size,
            random_state=iter_seed,
            shuffle=True,
        )
        model = UnimodalCNNModel(input_dim=train_ds.audio_dim, modality="audio", learning_rate=args.lr)
        model.train(X_tr, y_tr, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
        y_prob = model.predict(X_test)
        attention = None

    else:
        X_tr_a, X_val_a, X_tr_t, X_val_t, y_tr, y_val = train_test_split(
            train_ds.X_audio,
            train_ds.X_text,
            train_ds.y,
            test_size=args.val_size,
            random_state=iter_seed,
            shuffle=True,
        )
        model = AttentionFusionModel(audio_dim=train_ds.audio_dim, text_dim=train_ds.text_dim, learning_rate=args.lr)
        model.train(
            X_tr_a,
            X_tr_t,
            y_tr,
            X_val_a,
            X_val_t,
            y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        y_prob, attention = model.predict(test_ds.X_audio, test_ds.X_text)

    return model, y_prob, attention


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    run_dir_name = build_run_dir_name(args.model_type, args.text_model, args.audio_model)
    out_dir = base_dir / args.output_dir / args.model_type / run_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    need_audio = args.model_type in {"audio_cnn", "fusion_attention"}
    need_text = args.model_type in {"text_cnn", "fusion_attention"}

    train_ds = load_tabular_dataset(
        base_dir=base_dir,
        year=args.train_year,
        audio_model_name=args.audio_model if need_audio else None,
        text_model_name=args.text_model if need_text else None,
    )
    test_ds = load_tabular_dataset(
        base_dir=base_dir,
        year=args.test_year,
        audio_model_name=args.audio_model if need_audio else None,
        text_model_name=args.text_model if need_text else None,
    )

    prediction_frames = []
    for iteration in range(1, args.num_runs + 1):
        print(f"\n===== Iteration {iteration}/{args.num_runs} =====")
        model, y_prob, attention = train_predict_one_iteration(args, train_ds, test_ds, iteration)
        y_pred = binarize(y_prob, args.threshold)
        prediction_frames.append(
            prediction_dataframe(
                subjects=test_ds.subjects,
                y_prob=y_prob,
                y_pred=y_pred,
                label_cols=train_ds.label_cols,
                attention=attention,
                iteration=iteration,
            )
        )
        if args.save_models:
            suffix = f"run_{iteration:03d}"
            model_path = out_dir / f"model_{suffix}.keras"
            model.save(str(model_path))

    prediction_path = out_dir / f"all_iterations_test_{args.test_year}_multilabel_predictions.csv"
    save_all_iterations_prediction_csv(prediction_path, prediction_frames)
    print(f"\nSaved all iteration predictions to: {prediction_path}")


if __name__ == "__main__":
    main()
