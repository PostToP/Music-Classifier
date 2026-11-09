"""Prod model: one .npz archive (vocabularies, encoders, weights) run with numpy; no torch, sklearn or pandas."""

from __future__ import annotations

import json
import math

import numpy as np

from data.text_preprocessor import clean_text

DENSE_PATHS = ("title", "desc", "category", "duration", "post")
BLOCK_PREFIXES = {
    "title": "title_blocks",
    "desc": "description_blocks",
    "category": "category_blocks",
    "duration": "duration_blocks",
    "post": "post_blocks",
}

erf = np.vectorize(math.erf, otypes=[np.float32])


def apply_activation(name: str, values: np.ndarray) -> np.ndarray:
    """Apply a model.pytorch_model._activation activation with torch's defaults (exact GELU)."""
    if name == "relu":
        return np.maximum(values, 0)
    if name == "gelu":
        return values * 0.5 * (1 + erf(values / math.sqrt(2)))
    if name == "tanh":
        return np.tanh(values)
    message = f"Unsupported activation: {name}"
    raise ValueError(message)


def encode_tfidf(tokens: list[str], vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    """VectorizerTFIDF.encode: raw counts times idf, then l2-normalized; tokens outside the vocabulary are ignored."""
    counts = np.zeros(len(idf), dtype=np.float32)
    for token in tokens:
        if token in vocab:
            counts[vocab[token]] += 1
    weighted = counts * idf

    norm = np.linalg.norm(weighted)
    if norm == 0:
        return weighted
    return weighted / norm


class ModelWrapper:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.config = json.loads(str(arrays["config_json"]))
        self.title_vocab = {token: index for index, token in enumerate(arrays["title_vocab"].tolist())}
        self.title_idf = arrays["title_idf"]
        self.description_vocab = {token: index for index, token in enumerate(arrays["description_vocab"].tolist())}
        self.description_idf = arrays["description_idf"]
        self.category_indexes = {label: index for index, label in enumerate(arrays["category_labels"].tolist())}

        weight_names = [name for name in arrays if name.endswith((".weight", ".bias"))]
        weights = {name: arrays[name].astype(np.float32) for name in weight_names}
        self.blocks = {path: self._collect_blocks(path, weights) for path in DENSE_PATHS}
        self.output_weight = weights["output_layer.weight"]
        self.output_bias = weights["output_layer.bias"]

    def _collect_blocks(self, path: str, weights: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, str]]:
        prefix = BLOCK_PREFIXES[path]
        activations = self.config[f"{path}_activations"]
        return [
            (weights[f"{prefix}.{i}.linear.weight"], weights[f"{prefix}.{i}.linear.bias"], activation)
            for i, activation in enumerate(activations)
        ]

    @staticmethod
    def deserialize(location: str) -> ModelWrapper:
        with np.load(location, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return ModelWrapper(arrays)

    def warmup(self) -> None:
        self.predict("Warmup title", "Warmup description", ["Music"], 300)

    def _encode_categories(self, categories: list[str]) -> np.ndarray:
        """MultiLabelBinarizer.transform: one slot per training label; labels unseen in training are ignored."""
        encoded = np.zeros(len(self.category_indexes), dtype=np.float32)
        for category in categories:
            if category in self.category_indexes:
                encoded[self.category_indexes[category]] = 1
        return encoded

    @staticmethod
    def _encode_duration(duration: float) -> np.ndarray:
        """VectorizerRawNumber: the seconds as one float; training only saw finite, non-negative durations."""
        if not math.isfinite(duration) or duration < 0:
            message = f"Duration must be a finite, non-negative number of seconds, got {duration}"
            raise ValueError(message)
        return np.array([duration], dtype=np.float32)

    def _run_blocks(self, path: str, values: np.ndarray) -> np.ndarray:
        for weight, bias, activation in self.blocks[path]:
            values = apply_activation(activation, weight @ values + bias)
        return values

    def predict(self, title: str, description: str, categories: list[str], duration: float) -> float:
        title_vector = encode_tfidf(clean_text(title).split(), self.title_vocab, self.title_idf)
        description_vector = encode_tfidf(clean_text(description).split(), self.description_vocab, self.description_idf)

        combined = np.concatenate(
            [
                self._run_blocks("title", title_vector),
                self._run_blocks("desc", description_vector),
                self._run_blocks("category", self._encode_categories(categories)),
                self._run_blocks("duration", self._encode_duration(duration)),
            ],
        )
        logit = self.output_weight @ self._run_blocks("post", combined) + self.output_bias
        return float(np.exp(-np.logaddexp(0, -logit[0])))
