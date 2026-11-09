"""Hyperparameter search for the PyTorch model, scored on the prod holdout.

Trials train on videos outside the prod holdout and are scored on it: tuning on the older data alone made
the model worse on the videos prod actually sees.
"""

from functools import cache

import optuna
import torch
from sklearn.metrics import log_loss

from data.split_dataset import SEED, split_rows_by_video
from model.prod_holdout import (
    build_model_splits,
    load_rows_with_prod_predictions,
    select_default_language_rows,
)
from model.pytorch_model import build_config_from_params
from model.train import OptimizerSettings, fit_model, predict_probabilities

STUDY_NAME = "pytorch_tuning_prod_holdout_tfidf"
STORAGE = "sqlite:///optuna_study.db"
N_TRIALS = 400
VAL_FRACTION = 0.1

HIDDEN_UNITS = [8, 16, 32, 64, 128, 256]
ACTIVATIONS = ["relu", "gelu", "tanh"]
MAX_LAYERS_BY_DENSE_PATH = {"title": 2, "desc": 2, "category": 3, "duration": 3, "post": 2}


def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    for path, max_layers in MAX_LAYERS_BY_DENSE_PATH.items():
        n_layers = trial.suggest_int(f"n_{path}_layers", 0, max_layers)
        for i in range(n_layers):
            trial.suggest_categorical(f"{path}_units_{i}", HIDDEN_UNITS)
            trial.suggest_categorical(f"{path}_activation_{i}", ACTIVATIONS)
            trial.suggest_float(f"{path}_dropout_{i}", 0.0, 0.5)
            trial.suggest_float(f"{path}_l2_{i}", 1e-7, 1e-3, log=True)
    return trial.params


@cache
def build_tuning_splits() -> list:
    """Train and early-stopping val from videos outside the holdout, plus the holdout, one row per video, to score."""
    rows = load_rows_with_prod_predictions()
    holdout_videos = select_default_language_rows(rows.loc[rows["In Prod Holdout"]])
    training_pool = rows.loc[~rows["In Prod Holdout"]]

    train_rows, val_rows = split_rows_by_video(training_pool, VAL_FRACTION)
    return build_model_splits(train_rows, val_rows, holdout_videos)


def objective(trial: optuna.Trial) -> float:
    params = suggest_hyperparameters(trial)
    train_split, val_split, holdout_split = build_tuning_splits()
    input_dims = [inputs.shape[1] for inputs in train_split[0]]
    config = build_config_from_params(params, *input_dims)

    optimizer_settings = OptimizerSettings(params["learning_rate"], params["weight_decay"])
    model, _best_val_loss = fit_model(config, train_split, val_split, optimizer_settings, verbose=False)

    holdout_probabilities = predict_probabilities(model, holdout_split[0])
    return log_loss(holdout_split[1], holdout_probabilities, labels=[0, 1])


def fetch_best_params() -> dict:
    return optuna.load_study(study_name=STUDY_NAME, storage=STORAGE).best_params


def main() -> None:
    torch.manual_seed(SEED)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=STORAGE,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"Best trial: {study.best_trial.number}")
    print(f"  Log loss: {study.best_value:.5f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
