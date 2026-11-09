import json

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from model.pytorch_model import MusicClassifier, load_model_artifacts
from tokenizer.tokenizer_whitespace import TokenizerWhitespace

ARCHIVE_PATH = "model/model_wrapper.npz"
WEIGHT_DTYPE = np.float16


def extract_tfidf_tokens_and_idf(pipeline: Pipeline) -> tuple[np.ndarray, np.ndarray]:
    """Return the vocabulary ordered by column index and its idf weights from a whitespace + TF-IDF pipeline."""
    if not isinstance(pipeline.named_steps["tokenizer"], TokenizerWhitespace):
        message = f"ModelWrapper only splits on whitespace, got {pipeline.named_steps['tokenizer']!r}"
        raise TypeError(message)
    tfidf = pipeline.named_steps["vectorizer"].vectorizer
    tokens_by_index = sorted(tfidf.vocabulary_, key=tfidf.vocabulary_.get)
    return np.array(tokens_by_index, dtype=str), tfidf.idf_.astype(np.float32)


def build_archive_arrays(model: MusicClassifier, pipelines: dict[str, Pipeline]) -> dict[str, np.ndarray]:
    title_vocab, title_idf = extract_tfidf_tokens_and_idf(pipelines["title"])
    description_vocab, description_idf = extract_tfidf_tokens_and_idf(pipelines["description"])

    weights = {name: tensor.detach().cpu().numpy().astype(WEIGHT_DTYPE) for name, tensor in model.state_dict().items()}
    return {
        "config_json": np.array(json.dumps(model.config.to_dict())),
        "title_vocab": title_vocab,
        "title_idf": title_idf,
        "description_vocab": description_vocab,
        "description_idf": description_idf,
        "category_labels": pipelines["category"].named_steps["vectorizer"].vectorizer.classes_.astype(str),
        **weights,
    }


def create_model_wrapper() -> None:
    model, _config = load_model_artifacts("model/final_model.pt", "model/final_model_config.json")
    pipelines = {
        "title": joblib.load("model/pipelines/Title_pipeline.joblib"),
        "description": joblib.load("model/pipelines/Description_pipeline.joblib"),
        "category": joblib.load("model/pipelines/Categories_pipeline.joblib"),
    }
    np.savez_compressed(ARCHIVE_PATH, **build_archive_arrays(model, pipelines))
