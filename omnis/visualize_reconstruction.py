"""Visualise les valeurs PMU physiques et leur reconstruction OmniAnomaly.

Exemples (depuis la racine du dépôt) :

    python omnis/visualize_reconstruction.py
    python omnis/visualize_reconstruction.py --backend plotly --split train
"""

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from model import OmniAnomaly
from utils import (
    FEATURE_REPRESENTATION_VERSION,
    MinMaxScaler,
    get_data,
    inverse_transform_physical,
    phase_embedding_to_phasor,
    validate_checkpoint_model_dimension,
)


def load_config(path: Path) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file)
    return SimpleNamespace(**values)


def resolve_from_config(path: str | Path, config_path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else config_path.parent / path


def configured_value(arguments, config, name):
    """Donne la priorité à la CLI, puis à la configuration de visualisation."""
    argument_value = getattr(arguments, name)
    return argument_value if argument_value is not None else getattr(config, name)


def physical_reconstruction_indices(representation):
    """Retourne uniquement les grandeurs physiques réellement reconstructibles."""
    indices = [feature.physical_index for feature in representation.direct_features]
    for phasor in representation.phasors:
        indices.append(phasor.physical_amplitude_index)
        if phasor.includes_phase:
            indices.append(phasor.physical_phase_index)
    return indices


@torch.inference_mode()
def reconstruct_last_points(
    model,
    normalized_values,
    excludes,
    start: int,
    point_count: int,
    batch_size: int,
    n_z: int,
    device: torch.device,
):
    """Reconstruit chaque point demandé à partir de sa fenêtre causale."""
    window_length = model.window_length
    end = min(start + point_count, len(normalized_values))
    if start < window_length - 1:
        raise ValueError(
            f"start doit être au moins égal à {window_length - 1}"
        )
    if start >= end:
        raise ValueError("La plage demandée est vide")

    target_indices = torch.arange(start, end)
    offsets = torch.arange(-window_length + 1, 1)
    window_indices = target_indices.unsqueeze(1) + offsets
    valid_targets = ~torch.as_tensor(excludes)[window_indices].any(dim=1)
    valid_window_indices = window_indices[valid_targets]
    valid_positions = torch.nonzero(valid_targets).flatten()
    if not valid_targets.any():
        raise ValueError("Aucune fenêtre valide dans la plage demandée")
    reconstruction = torch.full(
        (end - start, normalized_values.shape[1]),
        float("nan"),
        dtype=torch.float32,
    )

    model.eval()
    for batch_start in range(0, len(valid_window_indices), batch_size):
        indices = valid_window_indices[batch_start:batch_start + batch_size]
        windows = normalized_values[indices].to(device)
        reconstructed = model.vae.reconstruct(
            windows,
            n_z=n_z,
            use_mean=True,
        ).mean(dim=0)
        positions = valid_positions[batch_start:batch_start + batch_size]
        reconstruction[positions] = reconstructed[:, -1].cpu()

    return target_indices, reconstruction


def physical_squared_errors(observed, reconstructed, representation):
    """Calcule la MSE physique, avec distance circulaire pour les phases."""
    residual = reconstructed - observed
    for phasor in representation.phasors:
        phase_index = phasor.physical_phase_index
        residual[:, phase_index] = (
            residual[:, phase_index] + 180.0
        ) % 360.0 - 180.0
    return np.nanmean(np.square(residual), axis=0)


def plot_matplotlib(x, observed, reconstructed, names, mse, output_path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=(12, max(3 * len(names), 6)),
        sharex=True,
        squeeze=False,
    )
    for index, name in enumerate(names):
        axis = axes[index, 0]
        axis.plot(
            x,
            observed[:, index],
            color="tab:blue",
            label="Observé",
        )
        axis.plot(
            x,
            reconstructed[:, index],
            color="tab:orange",
            linestyle="--",
            label="Reconstruit",
        )
        axis.set_title(f"{name} — MSE={mse[index]:.6g}")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1, 0].set_xlabel("Indice temporel")
    figure.suptitle(
        f"Reconstruction physique — MSE globale={np.nanmean(mse):.6g}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_plotly(x, observed, reconstructed, names, mse, output_path):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    figure = make_subplots(
        rows=len(names),
        cols=1,
        subplot_titles=[
            f"{name} — MSE={mse[index]:.6g}"
            for index, name in enumerate(names)
        ],
        shared_xaxes=True,
        vertical_spacing=min(0.04, 0.8 / len(names)),
    )
    for index, name in enumerate(names):
        figure.add_trace(
            go.Scatter(x=x, y=observed[:, index], name=f"{name} observé"),
            row=index + 1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=reconstructed[:, index],
                name=f"{name} reconstruit",
                line={"dash": "dash"},
            ),
            row=index + 1,
            col=1,
        )
    figure.update_layout(
        height=max(300 * len(names), 600),
        title=f"Reconstruction physique — MSE globale={np.nanmean(mse):.6g}",
    )
    figure.write_html(output_path, include_plotlyjs=True)


def main(arguments):
    config_path = Path(arguments.config).resolve()
    config = load_config(config_path)
    split = configured_value(arguments, config, "split")
    start = configured_value(arguments, config, "start")
    points = configured_value(arguments, config, "points")
    batch_size = configured_value(arguments, config, "batch_size")
    n_z = configured_value(arguments, config, "n_z")
    backend = configured_value(arguments, config, "backend")
    output = configured_value(arguments, config, "output")
    device_name = configured_value(arguments, config, "device")
    checkpoint = configured_value(arguments, config, "checkpoint")
    if points < 1 or batch_size < 1 or n_z < 1:
        raise ValueError("points, batch-size et n-z doivent être positifs")
    if split not in {"train", "test"}:
        raise ValueError("split doit valoir 'train' ou 'test'")
    if backend not in {"matplotlib", "plotly"}:
        raise ValueError("backend doit valoir 'matplotlib' ou 'plotly'")
    data_path = resolve_from_config(config.data_dir, config_path)
    train, test, representation = get_data(
        dataset=config.dataset,
        data_directory=data_path,
        max_train_size=config.max_train_size,
        max_test_size=config.max_test_size,
        train_start=config.train_start,
        print_log=False,
        return_metadata=True,
        exclude_zero_sequence_phase=config.exclude_zero_sequence_phase,
        data_column_order=config.data_column_order,
    )
    values, _, excludes = train if split == "train" else test
    if values is None:
        raise ValueError(f"La partie {split} est vide")
    config.x_dim = representation.model_dim

    checkpoint_path = resolve_from_config(checkpoint, config_path)
    device = torch.device(
        device_name
        if device_name is not None
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    validate_checkpoint_model_dimension(
        state,
        config.x_dim,
        expected_feature_names=representation.model_names,
        expected_representation_version=FEATURE_REPRESENTATION_VERSION,
    )
    model = OmniAnomaly(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    scaler = MinMaxScaler().load_state_dict(state["scaler_state_dict"])

    normalized_values = scaler.transform(values)
    indices, normalized_reconstruction = reconstruct_last_points(
        model=model,
        normalized_values=normalized_values,
        excludes=excludes,
        start=start,
        point_count=points,
        batch_size=batch_size,
        n_z=n_z,
        device=device,
    )
    selected_values = values[indices]
    observed = phase_embedding_to_phasor(selected_values, representation).numpy()
    reconstructed = inverse_transform_physical(
        normalized_reconstruction,
        scaler,
        representation,
    ).numpy()
    mse = physical_squared_errors(observed, reconstructed, representation)

    physical_indices = physical_reconstruction_indices(representation)
    observed = observed[:, physical_indices]
    reconstructed = reconstructed[:, physical_indices]
    mse = mse[physical_indices]
    names = [representation.original_names[index] for index in physical_indices]

    suffix = ".png" if backend == "matplotlib" else ".html"
    output = output or f"reconstruction_{split}{suffix}"
    output_path = Path(output)
    if output_path.suffix.lower() != suffix:
        raise ValueError(
            f"Le backend {backend} exige l'extension {suffix}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_function = (
        plot_matplotlib
        if backend == "matplotlib"
        else plot_plotly
    )
    plot_function(
        indices.numpy(),
        observed,
        reconstructed,
        names,
        mse,
        output_path,
    )
    print(f"Figure sauvegardée : {output_path.resolve()}")
    print(f"MSE globale physique : {np.nanmean(mse):.6g}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="omnis/visualize_config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--start", type=int)
    parser.add_argument("--points", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-z", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--backend",
        choices=("matplotlib", "plotly"),
    )
    parser.add_argument("--output")
    main(parser.parse_args())
