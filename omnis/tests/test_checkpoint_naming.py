import sys
from pathlib import Path
from types import SimpleNamespace


from omnis.main import best_checkpoint_filename  # noqa: E402
from omnis.utils import build_feature_representation  # noqa: E402


def _config(**overrides):
    values = {
        "normalization_mode": "nominal",
        "excluded_features": ["ROCOF"],
        "exclude_zero_sequence_phase": True,
        "window_length": 100,
        "z_dim": 3,
        "rnn_cell": "GRU",
        "rnn_num_hidden": 500,
        "decoder_sigma_mode": "fixed",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_best_checkpoint_name_describes_key_parameters_and_exclusions():
    representation = build_feature_representation(
        ["ROCOF", "FREQ", "V1.ANG", "V1.MAG"],
        excluded_features=["ROCOF"],
    )
    name = best_checkpoint_filename(_config(), representation)

    assert name.startswith("best_omni_anomaly__norm-nominal__features-4-")
    assert "__excl-rocof+zero-sequence-phase__" in name
    assert "__window-100__z-3__rnn-gru-500__decoder-fixed.pt" in name


def test_best_checkpoint_name_changes_when_a_key_parameter_changes():
    representation = build_feature_representation(
        ["ROCOF", "FREQ", "V1.ANG", "V1.MAG"],
        excluded_features=["ROCOF"],
    )
    baseline = best_checkpoint_filename(_config(), representation)

    assert best_checkpoint_filename(
        _config(normalization_mode="window_local"), representation
    ) != baseline
    assert best_checkpoint_filename(_config(z_dim=8), representation) != baseline
    assert best_checkpoint_filename(_config(window_length=50), representation) != baseline

