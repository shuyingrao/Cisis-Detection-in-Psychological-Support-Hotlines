"""TensorFlow/Keras models: unimodal CNN and text+audio attention fusion."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import (
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    Reshape,
)
from tensorflow.keras.optimizers import Adam


def _label_output_names(prefix: str = "pred") -> List[str]:
    return [f"{prefix}_mood", f"{prefix}_idea", f"{prefix}_plan", f"{prefix}_risk"]


def build_cnn_encoder(
    input_tensor: tf.Tensor,
    input_dim: int,
    latent_dim: int = 128,
    dropout: float = 0.1,
    name: str = "encoder",
) -> tf.Tensor:
    """Dimension-agnostic 1D CNN encoder for embedding vectors.

    The original scripts had separate 768/1024/1280/3072 CNNs. This encoder keeps the CNN idea
    but avoids hard-coded reshape/cropping rules by using same-padding Conv1D + global pooling.
    """
    x = Reshape((input_dim, 1), name=f"{name}_reshape")(input_tensor)
    x = Conv1D(32, kernel_size=7, strides=2, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = Dropout(dropout, name=f"{name}_drop1")(x)
    x = Conv1D(64, kernel_size=5, strides=2, padding="same", activation="relu", name=f"{name}_conv2")(x)
    x = Dropout(dropout, name=f"{name}_drop2")(x)
    x = Conv1D(128, kernel_size=3, strides=2, padding="same", activation="relu", name=f"{name}_conv3")(x)
    x = GlobalMaxPooling1D(name=f"{name}_gmp")(x)
    z = Dense(latent_dim, activation="relu", name=f"{name}_latent")(x)
    return z


def build_reconstruction_head(z: tf.Tensor, output_dim: int, name: str) -> tf.Tensor:
    """Lightweight AE auxiliary head used only during training."""
    x = Dense(256, activation="relu", name=f"{name}_recon_hidden")(z)
    return Dense(output_dim, activation=None, name=name)(x)


def build_multilabel_heads(z: tf.Tensor, prefix: str = "pred") -> List[tf.Tensor]:
    names = _label_output_names(prefix)
    return [Dense(1, activation="sigmoid", name=n)(z) for n in names]


def compile_unimodal_model(
    model: Model,
    learning_rate: float,
    reconstruction_weight: float,
    prefix: str = "pred",
) -> Model:
    pred_names = _label_output_names(prefix)
    loss = {name: "binary_crossentropy" for name in pred_names}
    loss["reconstruction"] = "mse"
    loss_weights = {name: 1.0 for name in pred_names}
    loss_weights["reconstruction"] = reconstruction_weight
    metrics = {name: ["accuracy"] for name in pred_names}
    model.compile(optimizer=Adam(learning_rate), loss=loss, loss_weights=loss_weights, metrics=metrics)
    return model


class UnimodalCNNModel:
    """Single-modality CNN for text or audio subject-level embeddings.

    modality examples:
        text  -> RoBERTa [CLS] 768 or GPT 3072
        audio -> wav2vec 1024, whisper 768/1024/1280, etc.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int = 4,
        modality: Literal["text", "audio"] = "text",
        learning_rate: float = 1e-3,
        latent_dim: int = 128,
        dropout: float = 0.1,
        reconstruction_weight: float = 0.8,
    ) -> None:
        if num_labels != 4:
            raise ValueError("This implementation uses four named binary heads. Set num_labels=4.")
        self.input_dim = input_dim
        self.modality = modality
        self.learning_rate = learning_rate
        self.model = self._build(latent_dim, dropout, reconstruction_weight)

    def _build(self, latent_dim: int, dropout: float, reconstruction_weight: float) -> Model:
        inputs = Input(shape=(self.input_dim,), name=f"input_{self.modality}")
        z = build_cnn_encoder(inputs, self.input_dim, latent_dim, dropout, name=f"{self.modality}_cnn")
        z = Dropout(dropout, name=f"{self.modality}_latent_dropout")(z)
        preds = build_multilabel_heads(z, prefix="pred")
        recon = build_reconstruction_head(z, self.input_dim, name="reconstruction")
        model = Model(inputs=inputs, outputs=[*preds, recon], name=f"{self.modality}_unimodal_cnn")
        return compile_unimodal_model(model, self.learning_rate, reconstruction_weight)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 150,
        batch_size: int = 32,
        patience: int = 30,
        log_dir: Optional[str] = None,
        verbose: int = 1,
    ):
        callbacks = [EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
        if log_dir:
            callbacks.append(TensorBoard(log_dir=log_dir))
        return self.model.fit(
            X_train,
            self._make_targets(X_train, y_train),
            validation_data=(X_val, self._make_targets(X_val, y_val)),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    @staticmethod
    def _make_targets(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "pred_mood": y[:, 0],
            "pred_idea": y[:, 1],
            "pred_plan": y[:, 2],
            "pred_risk": y[:, 3],
            "reconstruction": X,
        }

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        outputs = self.model.predict(X, batch_size=batch_size, verbose=0)
        return np.hstack(outputs[:4])

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 256):
        return self.model.evaluate(X, self._make_targets(X, y), batch_size=batch_size, verbose=0)

    def save(self, path: str) -> None:
        self.model.save(path)


class AttentionFusionModel:
    """Text + audio CNN encoders with scalar attention-gating fusion."""

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        learning_rate: float = 1e-3,
        latent_dim: int = 128,
        dropout: float = 0.2,
        reconstruction_weight: float = 0.5,
    ) -> None:
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.learning_rate = learning_rate
        self.model = self._build(latent_dim, dropout, reconstruction_weight)

    def _build(self, latent_dim: int, dropout: float, reconstruction_weight: float) -> Model:
        input_audio = Input(shape=(self.audio_dim,), name="input_audio")
        input_text = Input(shape=(self.text_dim,), name="input_text")

        z_audio = build_cnn_encoder(input_audio, self.audio_dim, latent_dim, dropout=0.1, name="audio")
        z_text = build_cnn_encoder(input_text, self.text_dim, latent_dim, dropout=0.1, name="text")

        concat = Concatenate(name="concat_audio_text")([z_audio, z_text])
        att = Dense(64, activation="relu", name="attention_hidden")(concat)
        att = Dropout(dropout, name="attention_dropout")(att)
        alpha = Dense(1, activation="sigmoid", name="attention_audio_alpha")(att)

        audio_weighted = Lambda(lambda x: x[0] * x[1], name="audio_weighted")([z_audio, alpha])
        text_weighted = Lambda(lambda x: x[0] * (1.0 - x[1]), name="text_weighted")([z_text, alpha])
        fused = Concatenate(name="fused_features")([audio_weighted, text_weighted])

        x = Dense(64, activation="relu", name="fusion_hidden")(fused)
        x = Dropout(dropout, name="fusion_dropout")(x)
        preds = build_multilabel_heads(x, prefix="pred")
        recon_audio = build_reconstruction_head(z_audio, self.audio_dim, name="recon_audio")
        recon_text = build_reconstruction_head(z_text, self.text_dim, name="recon_text")

        model = Model(
            inputs=[input_audio, input_text],
            outputs=[*preds, recon_audio, recon_text, alpha],
            name="attention_fusion_cnn",
        )
        loss = {name: "binary_crossentropy" for name in _label_output_names("pred")}
        loss.update({"recon_audio": "mse", "recon_text": "mse", "attention_audio_alpha": None})
        loss_weights = {name: 1.0 for name in _label_output_names("pred")}
        loss_weights.update({"recon_audio": reconstruction_weight, "recon_text": reconstruction_weight, "attention_audio_alpha": 0.0})
        metrics = {name: ["accuracy"] for name in _label_output_names("pred")}
        model.compile(optimizer=Adam(self.learning_rate), loss=loss, loss_weights=loss_weights, metrics=metrics)
        return model

    @staticmethod
    def _make_targets(X_audio: np.ndarray, X_text: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "pred_mood": y[:, 0],
            "pred_idea": y[:, 1],
            "pred_plan": y[:, 2],
            "pred_risk": y[:, 3],
            "recon_audio": X_audio,
            "recon_text": X_text,
            "attention_audio_alpha": np.zeros((len(X_audio), 1), dtype=np.float32),
        }

    def train(
        self,
        X_train_audio: np.ndarray,
        X_train_text: np.ndarray,
        y_train: np.ndarray,
        X_val_audio: np.ndarray,
        X_val_text: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 40,
        verbose: int = 1,
    ):
        callbacks = [EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
        return self.model.fit(
            {"input_audio": X_train_audio, "input_text": X_train_text},
            self._make_targets(X_train_audio, X_train_text, y_train),
            validation_data=(
                {"input_audio": X_val_audio, "input_text": X_val_text},
                self._make_targets(X_val_audio, X_val_text, y_val),
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    def predict(self, X_audio: np.ndarray, X_text: np.ndarray, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.model.predict({"input_audio": X_audio, "input_text": X_text}, batch_size=batch_size, verbose=0)
        y_prob = np.hstack(outputs[:4])
        attention = outputs[-1]
        return y_prob, attention

    def evaluate(self, X_audio: np.ndarray, X_text: np.ndarray, y: np.ndarray, batch_size: int = 256):
        return self.model.evaluate(
            {"input_audio": X_audio, "input_text": X_text},
            self._make_targets(X_audio, X_text, y),
            batch_size=batch_size,
            verbose=0,
        )

    def save(self, path: str) -> None:
        self.model.save(path)
