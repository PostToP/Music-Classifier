"""Labelled rows joined with the prod model's stored predictions, and feature encoding for splits over them."""

import copy

import numpy as np
import pandas as pd

from data.database import fetch_labelled_video_prod_predictions
from model.feature_extraction import FEATURES, transform_features

MODEL_INPUT_COLUMNS = ["Title", "Description", "Categories", "Duration"]


def load_rows_with_prod_predictions() -> pd.DataFrame:
    """Preprocessed rows plus "Default Language", "Prod Probability" and "In Prod Holdout".

    The holdout is every video prod predicted; prod stored each prediction before the label existed.
    """
    rows = pd.read_json("dataset/p2_dataset.json")
    prediction_rows = pd.DataFrame(
        fetch_labelled_video_prod_predictions(),
        columns=["YT ID", "Default Language", "Prod Probability"],
    )

    rows = rows.merge(prediction_rows, on="YT ID", how="inner", validate="many_to_one")
    rows["In Prod Holdout"] = rows["Prod Probability"].notna()
    return rows


def select_default_language_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per video: the default-language row, or the first row when dedup removed it."""
    is_default_language = rows["Language"] == rows["Default Language"]
    ordered_rows = rows.assign(is_default_language=is_default_language).sort_values(
        "is_default_language",
        ascending=False,
        kind="stable",
    )
    return ordered_rows.drop_duplicates("YT ID").drop(columns="is_default_language")


def build_model_splits(
    train_rows: pd.DataFrame,
    *evaluation_rows: pd.DataFrame,
) -> list[tuple[list[np.ndarray], np.ndarray]]:
    """Fit fresh feature pipelines on train_rows, then encode every split as (model inputs, labels)."""
    pipelines = copy.deepcopy(FEATURES)

    splits = []
    for index, rows in enumerate([train_rows, *evaluation_rows]):
        features = transform_features(rows[MODEL_INPUT_COLUMNS], fit=index == 0, pipelines=pipelines)
        inputs = [features[column] for column in MODEL_INPUT_COLUMNS]
        labels = rows["Is Music"].to_numpy(dtype=np.float32)
        splits.append((inputs, labels))
    return splits
