import pandas as pd
from sklearn.base import BaseEstimator


# scikit-learn >= 1.8 requires pipeline steps to inherit BaseEstimator (__sklearn_tags__).
class IVectorizer(BaseEstimator):
    output_dim: int = 0

    def train(self, text: list[str]) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        pass

    def get_dict(self) -> dict[str, int]:
        pass

    def fit(self, X: pd.Series, y: pd.Series) -> "IVectorizer":
        self.train(X)
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        encoded = self.encode_batch(X)
        return pd.DataFrame(encoded)

    def __repr__(self) -> str:
        pass

    def __sklearn_is_fitted__(self) -> bool:
        pass
