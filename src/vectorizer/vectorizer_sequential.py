import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer

from vectorizer.vectorizer import IVectorizer


class VectorizerSequential(IVectorizer):
    def __init__(
        self,
        vocab_size: int = 5000,
        max_sequence_length: int = 100,
        excluded_tokens_path: str | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.vectorizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        self.output_dim = max_sequence_length
        self.excluded_tokens_path = excluded_tokens_path

    def train(self, dataset: list[list[str]]) -> None:
        excluded_tokens = set()
        if self.excluded_tokens_path is not None:
            excluded_tokens = set(json.loads(Path(self.excluded_tokens_path).read_text()))
        kept_tokens = [[token for token in tokens if token not in excluded_tokens] for tokens in dataset]

        trainer = WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[PAD]"], show_progress=False)
        self.vectorizer.train_from_iterator(kept_tokens, trainer)

    def encode(self, text: list[str]) -> np.ndarray:
        sequence = self.vectorizer.encode(text, is_pretokenized=True)
        padded_ids = sequence.ids[: self.output_dim] + [0] * (self.output_dim - len(sequence.ids))
        return padded_ids[: self.output_dim]

    def encode_batch(self, texts: list[list[str]]) -> np.ndarray:
        sequence = self.vectorizer.encode_batch(texts, is_pretokenized=True)
        padded_ids = [s.ids[: self.output_dim] + [0] * (self.output_dim - len(s.ids)) for s in sequence]
        return np.array([ids[: self.output_dim] for ids in padded_ids])

    def get_dict(self) -> dict[str, int]:
        return self.vectorizer.__dict__()

    def __repr__(self) -> str:
        return f"VectorizerSequential_{self.vocab_size}_{self.output_dim}"

    def __sklearn_is_fitted__(self) -> bool:
        return self.vectorizer.get_vocab_size() > 0
