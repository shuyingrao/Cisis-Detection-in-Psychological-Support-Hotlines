"""Prediction post-processing utilities.

Test-performance reporting was intentionally removed from the training scripts.
This module keeps only the thresholding helper needed to convert probabilities
into multilabel binary predictions.
"""

from __future__ import annotations

import numpy as np


def binarize(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert multilabel probabilities to 0/1 predictions."""
    return (np.asarray(y_prob) >= threshold).astype(int)
