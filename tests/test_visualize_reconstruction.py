import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "omnis"))

from utils import build_feature_representation  # noqa: E402
from visualize_reconstruction import (  # noqa: E402
    configured_value,
    physical_reconstruction_indices,
)


def _columns():
    return ["ROCOF", "FREQ"] + [
        name
        for quantity in ("VA_V0", "VA_V1", "VA_V2", "IA_I0", "IA_I1", "IA_I2")
        for name in (f"VIGY_1:{quantity}_ANG", f"VIGY_1:{quantity}_MAG")
    ]


def test_visualization_uses_reduced_representation_without_fake_phase_plots():
    representation = build_feature_representation(
        _columns(), exclude_zero_sequence_phase=True
    )
    indices = physical_reconstruction_indices(representation)

    assert indices == [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
    assert 2 not in indices  # phase V0
    assert 8 not in indices  # phase I0
    assert 3 in indices and 9 in indices  # amplitudes V0 et I0


def test_cli_values_override_visualization_config():
    arguments = SimpleNamespace(points=None, backend="plotly")
    config = SimpleNamespace(points=1000, backend="matplotlib")
    assert configured_value(arguments, config, "points") == 1000
    assert configured_value(arguments, config, "backend") == "plotly"


def test_visualization_config_matches_training_representation_and_model():
    root = Path(__file__).parents[1]
    with (root / "omnis/config.yaml").open(encoding="utf-8") as file:
        training = yaml.safe_load(file)
    with (root / "omnis/visualize_config.yaml").open(encoding="utf-8") as file:
        visualization = yaml.safe_load(file)

    matching_keys = (
        "dataset", "data_dir", "data_column_order", "exclude_zero_sequence_phase",
        "use_connected_z_q", "use_connected_z_p", "z_dim",
        "rnn_num_hidden", "window_length", "dense_dim", "std_epsilon",
        "decoder_sigma_mode", "decoder_fixed_sigma", "get_score_on_dim",
        "train_start", "max_train_size", "max_test_size",
    )
    assert {key: visualization[key] for key in matching_keys} == {
        key: training[key] for key in matching_keys
    }
