from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from model.pytorch_model import (
    MusicClassifier,
    TorchModelConfig,
    build_config_from_params,
    print_model_summary,
    save_model_artifacts,
)

logger = logging.getLogger("experiment")

TOTAL_EPOCHS = 5000
BATCH_SIZE = 1024
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.0001
PREDICTION_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class OptimizerSettings:
    learning_rate: float
    weight_decay: float


class SplitDataset(Dataset):
    """A whole split held on the training device, so batches are sliced there instead of copied from host memory."""

    def __init__(self, inputs: list[np.ndarray], labels: np.ndarray, device: torch.device) -> None:
        self.title_vectors = torch.as_tensor(inputs[0], dtype=torch.float32, device=device)
        self.description_vectors = torch.as_tensor(inputs[1], dtype=torch.float32, device=device)
        self.categories = torch.as_tensor(inputs[2], dtype=torch.float32, device=device)
        self.duration = torch.as_tensor(inputs[3], dtype=torch.float32, device=device)
        self.labels = torch.as_tensor(labels, dtype=torch.float32, device=device).view(-1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.title_vectors[index],
            self.description_vectors[index],
            self.categories[index],
            self.duration[index],
            self.labels[index],
        )


def load_split(path: str) -> tuple[list[np.ndarray], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    inputs = [
        data["YT_ID"],
        data["Title"],
        data["Description"],
        data["Categories"],
        data["Duration"],
    ]
    labels = data["Labels"]
    return inputs, labels


def _build_batch_loader(dataset: SplitDataset, *, shuffle: bool) -> DataLoader:
    # Indexing whole batches at once; per-item __getitem__ plus collate left the GPU idle.
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=BatchSampler(sampler, BATCH_SIZE, drop_last=False), batch_size=None)


def _accuracy(probabilities: Tensor, labels: Tensor) -> float:
    predictions = (probabilities >= PREDICTION_THRESHOLD).float()
    return (predictions == labels).float().mean().item()


def _evaluate(
    model: MusicClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_items = 0

    with torch.no_grad():
        for title_batch, description_batch, category_batch, duration_batch, label_batch in loader:
            logits = model(title_batch, description_batch, category_batch, duration_batch)
            loss = criterion(logits, label_batch) + model.regularization_loss()
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= PREDICTION_THRESHOLD).float()

            batch_size = label_batch.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == label_batch).sum().item()
            total_items += batch_size

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def _train_one_epoch(
    model: MusicClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_items = 0

    for title_batch, description_batch, category_batch, duration_batch, label_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(title_batch, description_batch, category_batch, duration_batch)
        loss = criterion(logits, label_batch) + model.regularization_loss()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        probabilities = torch.sigmoid(logits)
        batch_size = label_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += _accuracy(probabilities, label_batch) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def _make_example_inputs(train_inputs: list[np.ndarray], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.zeros((1, train_inputs[0].shape[1]), dtype=torch.float32, device=device),
        torch.zeros((1, train_inputs[1].shape[1]), dtype=torch.float32, device=device),
        torch.zeros((1, train_inputs[2].shape[1]), dtype=torch.float32, device=device),
        torch.zeros((1, train_inputs[3].shape[1]), dtype=torch.float32, device=device),
    )


def _format_epoch_line(
    epoch: int,
    train_accuracy: float,
    train_loss: float,
    val_accuracy: float,
    val_loss: float,
    learning_rate: float,
) -> str:
    return (
        f"{epoch}/{TOTAL_EPOCHS} ━━━━━━━━━━━━━━━━━━━━ "
        f"- accuracy: {train_accuracy:.4f} - loss: {train_loss:.4f} "
        f"- val_accuracy: {val_accuracy:.4f} - val_loss: {val_loss:.4f} "
        f"- learning_rate: {learning_rate:.4e}"
    )


def fit_model(
    config: TorchModelConfig,
    train_df: tuple[list[np.ndarray], np.ndarray],
    val_df: tuple[list[np.ndarray], np.ndarray],
    optimizer_settings: OptimizerSettings,
    *,
    verbose: bool = True,
) -> tuple[MusicClassifier, float]:
    """Train with early stopping on validation loss; returns the best-epoch model and that loss."""
    train_inputs = train_df[0]
    model = MusicClassifier(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = _build_batch_loader(SplitDataset(train_inputs, train_df[1], device), shuffle=True)
    val_loader = _build_batch_loader(SplitDataset(val_df[0], val_df[1], device), shuffle=False)

    if verbose:
        print_model_summary(model, _make_example_inputs(train_inputs, device))
        logger.info("Compiling model")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_settings.learning_rate,
        weight_decay=optimizer_settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        train_loss, train_accuracy = _train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = _evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]

        if verbose:
            print(f"Epoch {epoch}/{TOTAL_EPOCHS}")
            print(
                _format_epoch_line(
                    epoch,
                    train_accuracy,
                    train_loss,
                    val_accuracy,
                    val_loss,
                    learning_rate,
                ),
            )
            logger.info(
                "Epoch %s: train_loss=%.6f train_acc=%.4f val_loss=%.6f val_acc=%.4f lr=%.6e",
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                learning_rate,
            )

        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping after %s epochs", epoch)
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def predict_probabilities(model: MusicClassifier, inputs: list[np.ndarray]) -> np.ndarray:
    labels_placeholder = np.zeros(len(inputs[0]), dtype=np.float32)
    device = next(model.parameters()).device
    loader = _build_batch_loader(SplitDataset(inputs, labels_placeholder, device), shuffle=False)
    model.eval()

    probability_batches = []
    with torch.no_grad():
        for title_batch, description_batch, category_batch, duration_batch, _ in loader:
            logits = model(title_batch, description_batch, category_batch, duration_batch)
            probability_batches.append(torch.sigmoid(logits).cpu().numpy().flatten())

    return np.concatenate(probability_batches)


def train_and_evaluate(
    train_df: tuple[list[np.ndarray], np.ndarray],
    test_df: tuple[list[np.ndarray], np.ndarray],
    val_df: tuple[list[np.ndarray], np.ndarray],
    params: dict,
) -> tuple[MusicClassifier, float, float, dict[str, int]]:
    train_inputs = train_df[0]

    input_dims = [inputs.shape[1] for inputs in train_inputs]
    config = build_config_from_params(params, *input_dims)
    optimizer_settings = OptimizerSettings(params["learning_rate"], params["weight_decay"])
    model, _best_val_loss = fit_model(config, train_df, val_df, optimizer_settings)

    device = next(model.parameters()).device
    test_loader = _build_batch_loader(SplitDataset(test_df[0], test_df[1], device), shuffle=False)
    test_loss, test_accuracy = _evaluate(model, test_loader, torch.nn.BCEWithLogitsLoss())
    print(f"{len(test_loader)}/{len(test_loader)} ━━━━━━━━━━━━━━━━━━━━")
    print(f"Validation accuracy: {test_accuracy}, loss: {test_loss}")
    logger.info("Validation accuracy: %s, loss: %s", test_accuracy, test_loss)

    return model, test_loss, test_accuracy, asdict(config)


def create_model(params: dict) -> None:
    train_df = load_split("dataset/p4_train.npz")
    val_df = load_split("dataset/p4_val.npz")
    test_df = load_split("dataset/p4_test.npz")

    train_df = (train_df[0][1:], train_df[1])
    val_df = (val_df[0][1:], val_df[1])
    test_df = (test_df[0][1:], test_df[1])

    model, _loss, _acc, config = train_and_evaluate(
        train_df,
        test_df,
        val_df,
        params,
    )

    logger.info("Model configuration: %s", config)

    save_model_artifacts(model, model.config, "model/final_model.pt", "model/final_model_config.json")
