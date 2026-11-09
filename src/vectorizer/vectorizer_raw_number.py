import numpy as np

from vectorizer.vectorizer import IVectorizer


class VectorizerRawNumber(IVectorizer):
    """Each number as one float feature, unscaled."""

    output_dim = 1

    def encode(self, number: float) -> np.ndarray:
        return np.array([number], dtype=np.float32)

    def encode_batch(self, numbers: list[float]) -> np.ndarray:
        return np.asarray(numbers, dtype=np.float32).reshape(-1, 1)

    def __repr__(self) -> str:
        return "VectorizerRawNumber"

    def __sklearn_is_fitted__(self) -> bool:
        return True
