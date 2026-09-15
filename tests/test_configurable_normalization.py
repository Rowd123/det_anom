import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "omnis"))

from utils import (  # noqa: E402
    WindowLocalNormalizer,
    build_feature_representation,
    build_normalizer,
    phase_embedding_to_phasor,
    phasor_to_phase_embedding,
)


def _columns():
    return ["ROCOF", "FREQ", "V1.ANG", "V1.MAG"]


def test_exclusions_apply_to_physical_names_and_embedding_components():
    representation = build_feature_representation(
        _columns(), excluded_features=["ROCOF", "V1.ANG"]
    )
    assert representation.model_names == ("freq", "v1_mag")
    assert set(representation.excluded_model_names) == {
        "rocof", "v1_phase_cos", "v1_phase_sin",
    }
    embedded = phasor_to_phase_embedding(
        np.array([[0.1, 50.0, 30.0, 240000.0]]), representation
    )
    np.testing.assert_allclose(embedded, [[50.0, 240000.0]])


def test_nominal_keeps_phase_geometry_and_round_trips():
    representation = build_feature_representation(_columns())
    normalizer = build_normalizer(
        "nominal",
        representation,
        {
            "ROCOF": {"center": 0.0, "scale": 1.0},
            "FREQ": {"center": 50.0, "scale": 0.5},
            "V1.MAG": {"center": 0.0, "scale": 240000.0},
        },
    )
    embedded = phasor_to_phase_embedding(
        np.array([[0.2, 50.5, 30.0, 120000.0]]), representation
    )
    normalized, context = normalizer.transform(embedded)
    indices = {name: index for index, name in enumerate(representation.model_names)}
    torch.testing.assert_close(normalized[0, indices["freq"]], torch.tensor(1.0))
    torch.testing.assert_close(
        normalized[0, indices["v1_phase_cos"]],
        torch.tensor(np.cos(np.deg2rad(30)), dtype=torch.float32),
    )
    restored = normalizer.inverse_transform(normalized, context)
    torch.testing.assert_close(restored, torch.as_tensor(embedded, dtype=torch.float32))
    physical = phase_embedding_to_phasor(restored, representation)
    torch.testing.assert_close(
        physical, torch.tensor([[0.2, 50.5, 30.0, 120000.0]])
    )


def test_window_local_has_independent_contexts_and_constant_feature_is_safe():
    windows = torch.tensor([
        [[1.0, 7.0], [3.0, 7.0]],
        [[10.0, -2.0], [20.0, -2.0]],
    ])
    normalizer = WindowLocalNormalizer()
    normalized, context = normalizer.transform(windows)
    assert context.minimum.shape == (2, 1, 2)
    assert context.maximum.shape == (2, 1, 2)
    torch.testing.assert_close(normalized[:, :, 1], torch.zeros(2, 2))
    torch.testing.assert_close(
        normalizer.inverse_transform(normalized, context), windows
    )

