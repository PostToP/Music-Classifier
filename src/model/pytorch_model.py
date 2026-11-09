from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

DENSE_PATHS = ("title", "desc", "category", "duration", "post")


@dataclass(slots=True)
class TorchModelConfig:
    title_input_dim: int
    desc_input_dim: int
    category_input_dim: int
    duration_input_dim: int
    title_hidden_dims: tuple[int, ...] = ()
    title_activations: tuple[str, ...] = ()
    title_dropouts: tuple[float, ...] = ()
    title_l2: tuple[float, ...] = ()
    desc_hidden_dims: tuple[int, ...] = ()
    desc_activations: tuple[str, ...] = ()
    desc_dropouts: tuple[float, ...] = ()
    desc_l2: tuple[float, ...] = ()
    category_hidden_dims: tuple[int, ...] = ()
    category_activations: tuple[str, ...] = ()
    category_dropouts: tuple[float, ...] = ()
    category_l2: tuple[float, ...] = ()
    duration_hidden_dims: tuple[int, ...] = ()
    duration_activations: tuple[str, ...] = ()
    duration_dropouts: tuple[float, ...] = ()
    duration_l2: tuple[float, ...] = ()
    post_hidden_dims: tuple[int, ...] = ()
    post_activations: tuple[str, ...] = ()
    post_dropouts: tuple[float, ...] = ()
    post_l2: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TorchModelConfig:
        values = {}
        for field in fields(cls):
            value = data[field.name]
            values[field.name] = tuple(value) if isinstance(value, list) else value
        return cls(**values)


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    message = f"Unsupported activation: {name}"
    raise ValueError(message)


class DenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, dropout: float, l2: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = _activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.l2 = l2

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.linear(inputs)
        outputs = self.activation(outputs)
        return self.dropout(outputs)

    def regularization_loss(self) -> Tensor:
        if self.l2 <= 0:
            return self.linear.weight.new_zeros(())
        return self.linear.weight.pow(2).sum() * self.l2


class MusicClassifier(nn.Module):
    def __init__(self, config: TorchModelConfig) -> None:
        super().__init__()
        self.config = config

        self.title_blocks, title_output_dim = self._build_blocks(
            input_dim=config.title_input_dim,
            hidden_dims=config.title_hidden_dims,
            activations=config.title_activations,
            dropouts=config.title_dropouts,
            l2_values=config.title_l2,
        )
        self.description_blocks, description_output_dim = self._build_blocks(
            input_dim=config.desc_input_dim,
            hidden_dims=config.desc_hidden_dims,
            activations=config.desc_activations,
            dropouts=config.desc_dropouts,
            l2_values=config.desc_l2,
        )
        self.category_blocks, category_output_dim = self._build_blocks(
            input_dim=config.category_input_dim,
            hidden_dims=config.category_hidden_dims,
            activations=config.category_activations,
            dropouts=config.category_dropouts,
            l2_values=config.category_l2,
        )
        self.duration_blocks, duration_output_dim = self._build_blocks(
            input_dim=config.duration_input_dim,
            hidden_dims=config.duration_hidden_dims,
            activations=config.duration_activations,
            dropouts=config.duration_dropouts,
            l2_values=config.duration_l2,
        )

        combined_dim = title_output_dim + description_output_dim + category_output_dim + duration_output_dim
        self.post_blocks, post_output_dim = self._build_blocks(
            input_dim=combined_dim,
            hidden_dims=config.post_hidden_dims,
            activations=config.post_activations,
            dropouts=config.post_dropouts,
            l2_values=config.post_l2,
        )
        self.output_layer = nn.Linear(post_output_dim, 1)

    @staticmethod
    def _build_blocks(
        input_dim: int,
        hidden_dims: tuple[int, ...],
        activations: tuple[str, ...],
        dropouts: tuple[float, ...],
        l2_values: tuple[float, ...],
    ) -> tuple[nn.ModuleList, int]:
        blocks = nn.ModuleList()
        current_dim = input_dim
        for hidden_dim, activation, dropout, l2 in zip(hidden_dims, activations, dropouts, l2_values, strict=True):
            blocks.append(DenseBlock(current_dim, hidden_dim, activation, dropout, l2))
            current_dim = hidden_dim
        return blocks, current_dim

    def _run_blocks(self, inputs: Tensor, blocks: nn.ModuleList) -> Tensor:
        outputs = inputs
        for block in blocks:
            outputs = block(outputs)
        return outputs

    def regularization_loss(self) -> Tensor:
        all_blocks = [
            *self.title_blocks,
            *self.description_blocks,
            *self.category_blocks,
            *self.duration_blocks,
            *self.post_blocks,
        ]
        losses = [block.regularization_loss() for block in all_blocks]
        if not losses:
            return self.output_layer.weight.new_zeros(())
        return torch.stack(losses).sum()

    def forward(self, title: Tensor, description: Tensor, categories: Tensor, duration: Tensor) -> Tensor:
        title_features = self._run_blocks(title.float(), self.title_blocks)
        description_features = self._run_blocks(description.float(), self.description_blocks)
        category_features = self._run_blocks(categories.float(), self.category_blocks)
        duration_features = self._run_blocks(duration.float(), self.duration_blocks)
        combined = torch.cat([title_features, description_features, category_features, duration_features], dim=1)
        return self.output_layer(self._run_blocks(combined, self.post_blocks))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def print_model_summary(model: MusicClassifier, example_inputs: tuple[Tensor, Tensor, Tensor, Tensor]) -> None:
    records: list[tuple[str, str, str, int]] = []
    hooks = []

    def register_hook(name: str, module: nn.Module) -> None:
        if any(module.children()):
            return

        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], outputs: Tensor) -> None:
            params = sum(parameter.numel() for parameter in _module.parameters())
            shape = str(tuple(outputs.shape)) if hasattr(outputs, "shape") else "-"
            records.append((name, _module.__class__.__name__, shape, params))

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if name:
            register_hook(name, module)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(*example_inputs)

    for hook in hooks:
        hook.remove()

    if was_training:
        model.train()

    print(model.__class__.__name__)
    print("=" * 80)
    print(f"{'Layer (type)':<35} {'Output Shape':<22} {'Param #':>12}")
    print("-" * 80)
    for layer_name, layer_type, output_shape, params in records:
        layer_label = f"{layer_name} ({layer_type})"
        print(f"{layer_label:<35} {output_shape:<22} {params:,}")
    total_params, trainable_params = count_parameters(model)
    print("-" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print("Non-trainable params: 0")


def build_config_from_params(
    params: dict,
    title_input_dim: int,
    desc_input_dim: int,
    category_input_dim: int,
    duration_input_dim: int,
) -> TorchModelConfig:
    """Config from flat Optuna params: n_<path>_layers, then <path>_units/_activation/_dropout/_l2_<i>."""
    dense_path_fields = {}
    for path in DENSE_PATHS:
        layer_indexes = range(params[f"n_{path}_layers"])
        dense_path_fields[f"{path}_hidden_dims"] = tuple(params[f"{path}_units_{i}"] for i in layer_indexes)
        dense_path_fields[f"{path}_activations"] = tuple(params[f"{path}_activation_{i}"] for i in layer_indexes)
        dense_path_fields[f"{path}_dropouts"] = tuple(params[f"{path}_dropout_{i}"] for i in layer_indexes)
        dense_path_fields[f"{path}_l2"] = tuple(params[f"{path}_l2_{i}"] for i in layer_indexes)

    return TorchModelConfig(
        title_input_dim=title_input_dim,
        desc_input_dim=desc_input_dim,
        category_input_dim=category_input_dim,
        duration_input_dim=duration_input_dim,
        **dense_path_fields,
    )


def save_model_artifacts(model: MusicClassifier, config: TorchModelConfig, model_path: str, config_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    Path(config_path).write_text(json.dumps(config.to_dict(), indent=2))


def load_model_artifacts(
    model_path: str,
    config_path: str,
    map_location: str = "cpu",
) -> tuple[MusicClassifier, TorchModelConfig]:
    config = TorchModelConfig.from_dict(json.loads(Path(config_path).read_text()))
    model = MusicClassifier(config)
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()
    return model, config
