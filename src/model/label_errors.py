"""Search for mislabelled videos: out-of-fold predictions for every row, ranked by disagreement.

Each video is predicted by a model that never trained on it (GroupKFold by video, best model.tuning params), so a
confident disagreement with the label points at the label rather than at memorization.
Run: python src/cli.py labelerrors
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from data.split_dataset import SEED, split_rows_by_video
from model.prod_holdout import build_model_splits
from model.pytorch_model import build_config_from_params
from model.train import OptimizerSettings, fit_model, predict_probabilities
from model.tuning import VAL_FRACTION, fetch_best_params

FOLD_COUNT = 5
PRINTED_CANDIDATE_COUNT = 80
REPORT_PATH = Path("reports/label_error_candidates.csv")


def predict_out_of_fold_probabilities(rows: pd.DataFrame, params: dict) -> np.ndarray:
    probabilities = np.full(len(rows), np.nan, dtype=np.float32)
    folds = GroupKFold(n_splits=FOLD_COUNT).split(rows, groups=rows["YT ID"])
    for fold_number, (train_index, held_out_index) in enumerate(folds, start=1):
        train_pool = rows.iloc[train_index].reset_index(drop=True)
        train_rows, val_rows = split_rows_by_video(train_pool, VAL_FRACTION)
        held_out_rows = rows.iloc[held_out_index].reset_index(drop=True)
        train_split, val_split, held_out_split = build_model_splits(train_rows, val_rows, held_out_rows)

        input_dims = [inputs.shape[1] for inputs in train_split[0]]
        config = build_config_from_params(params, *input_dims)
        optimizer_settings = OptimizerSettings(params["learning_rate"], params["weight_decay"])
        model, _best_val_loss = fit_model(config, train_split, val_split, optimizer_settings, verbose=False)

        probabilities[held_out_index] = predict_probabilities(model, held_out_split[0])
        print(f"Fold {fold_number}/{FOLD_COUNT} done")
    return probabilities


def summarize_videos(rows: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    """One row per video: its label, the mean out-of-fold probability over its language rows, and what it shows."""
    scored_rows = rows.assign(probability=probabilities)
    video_summaries = scored_rows.groupby("YT ID").agg(
        is_music=("Is Music", "first"),
        mean_probability=("probability", "mean"),
        language_rows=("probability", "size"),
        channel=("Channel Name", "first"),
        duration=("Duration", "first"),
        categories=("Categories", "first"),
        title=("Title", "first"),
        description_start=("Description", lambda descriptions: descriptions.iloc[0][:150]),
    )
    labels = video_summaries["is_music"].astype(float)
    video_summaries["disagreement"] = (labels - video_summaries["mean_probability"]).abs()
    return video_summaries.sort_values("disagreement", ascending=False).reset_index()


def find_label_error_candidates() -> None:
    torch.manual_seed(SEED)
    rows = pd.read_json("dataset/p2_dataset.json")
    params = fetch_best_params()

    probabilities = predict_out_of_fold_probabilities(rows, params)
    video_summaries = summarize_videos(rows, probabilities)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    video_summaries.to_csv(REPORT_PATH, index=False)
    for threshold in [0.5, 0.8, 0.95]:
        print(f"Videos with disagreement > {threshold}: {(video_summaries['disagreement'] > threshold).sum()}")
    columns = ["YT ID", "is_music", "mean_probability", "channel", "duration", "title"]
    print(video_summaries[columns].head(PRINTED_CANDIDATE_COUNT).to_string(index=False))
