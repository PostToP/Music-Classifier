import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from vectorizer.vectorizer import IVectorizer


# A module-level function, not a lambda, so the fitted pipeline pickles with joblib.
def return_tokens_unchanged(tokens: list[str]) -> list[str]:
    return tokens


class VectorizerTFIDF(IVectorizer):
    def __init__(self, max_features: int = 5000, excluded_tokens_path: str | None = None) -> None:
        self.max_features = max_features
        self.excluded_tokens_path = excluded_tokens_path
        self.vectorizer = TfidfVectorizer(
            tokenizer=return_tokens_unchanged,
            preprocessor=return_tokens_unchanged,
            token_pattern=None,
            lowercase=False,
            max_features=max_features,
        )
        self.output_dim = max_features

    def train(self, dataset: list[list[str]]) -> None:
        excluded_tokens = set()
        if self.excluded_tokens_path is not None:
            excluded_tokens = set(json.loads(Path(self.excluded_tokens_path).read_text()))
        kept_tokens = [[token for token in tokens if token not in excluded_tokens] for tokens in dataset]

        self.vectorizer.fit(kept_tokens)

    def encode(self, text: list[str]) -> np.ndarray:
        text = [text]
        return self.vectorizer.transform(text).astype(np.float32).toarray()[0]

    def encode_batch(self, texts: list[list[str]]) -> np.ndarray:
        return self.vectorizer.transform(texts).astype(np.float32).toarray()

    def get_dict(self) -> dict[str, int]:
        voc = self.vectorizer.vocabulary_
        voc = dict(sorted(voc.items(), key=lambda item: item[1]))
        return voc

    def __repr__(self) -> str:
        return f"VectorizerTFIDF_{len(self.vectorizer.vocabulary_)}"

    def __sklearn_is_fitted__(self) -> bool:
        return len(self.vectorizer.vocabulary_) > 0
