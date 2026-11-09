import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from data.preprocess import ARTIST_TOKENS_PATH
from tokenizer.tokenizer_whitespace import TokenizerWhitespace
from vectorizer.vectorizer_label import VectorizerLabel
from vectorizer.vectorizer_raw_number import VectorizerRawNumber
from vectorizer.vectorizer_tfidf import VectorizerTFIDF

logger = logging.getLogger("experiment")


FEATURES = {
    "Title": Pipeline(
        [
            ("tokenizer", TokenizerWhitespace()),
            ("vectorizer", VectorizerTFIDF(4000, excluded_tokens_path=ARTIST_TOKENS_PATH)),
        ],
    ),
    "Description": Pipeline(
        [
            ("tokenizer", TokenizerWhitespace()),
            ("vectorizer", VectorizerTFIDF(4000, excluded_tokens_path=ARTIST_TOKENS_PATH)),
        ],
    ),
    "Categories": Pipeline(
        [
            ("vectorizer", VectorizerLabel()),
        ],
    ),
    "Duration": Pipeline(
        [
            ("vectorizer", VectorizerRawNumber()),
        ],
    ),
}


def transform_features(df, fit=False, pipelines: dict[str, Pipeline] = FEATURES) -> dict:
    outputs = {}
    for col_name, pipeline in pipelines.items():
        if fit:
            outputs[col_name] = pipeline.fit_transform(df[col_name]).to_numpy()
        else:
            outputs[col_name] = pipeline.transform(df[col_name]).to_numpy()

    passthroug_cols = [c for c in df.columns if c not in outputs]
    for col_name in passthroug_cols:
        outputs[col_name] = df[col_name].to_numpy()

    return outputs


def save_split(path, X, y):
    np.savez_compressed(
        path,
        YT_ID=X["YT ID"],
        Title=X["Title"],
        Description=X["Description"],
        Categories=X["Categories"],
        Duration=X["Duration"],
        Labels=y,
    )


def prepare_datasets_and_pipelines() -> None:
    train_df = pd.read_json("dataset/p3_dataset_split.json")
    val_df = pd.read_json("dataset/p3_dataset_validation.json")
    test_df = pd.read_json("dataset/p3_dataset_test.json")

    X_train = transform_features(train_df, fit=True)
    X_val = transform_features(val_df, fit=False)
    X_test = transform_features(test_df, fit=False)

    save_split("dataset/p4_train.npz", X_train, train_df["Is Music"].to_numpy())
    save_split("dataset/p4_val.npz", X_val, val_df["Is Music"].to_numpy())
    save_split("dataset/p4_test.npz", X_test, test_df["Is Music"].to_numpy())

    for name, pipeline in FEATURES.items():
        joblib.dump(pipeline, f"model/pipelines/{name}_pipeline.joblib")
