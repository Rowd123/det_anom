import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "omnis"))

from utils import (  # noqa: E402
    MinMaxScaler,
    build_feature_representation,
    inverse_transform_physical,
    phase_embedding_to_phasor,
    phasor_to_phase_embedding,
    validate_checkpoint_model_dimension,
)


def representation():
    return build_feature_representation(
        ["FREQ", "DFDT"]
        + [name for i in range(6) for name in (f"P{i}MAG", f"P{i}ANG")]
    )


def physical_row(amplitude=1.0, phase=30.0):
    return np.array([[50.0, 0.1] + [v for _ in range(6) for v in (amplitude, phase)]])


def test_mapping_and_round_trip_numpy():
    mapping = representation()
    embedded = phasor_to_phase_embedding(physical_row(), mapping)
    assert mapping.physical_dim == 14
    assert mapping.model_dim == 20
    assert embedded.shape == (1, 20)
    first = mapping.phasors[0]
    assert embedded[0, first.model_amplitude_index] == 1.0
    np.testing.assert_allclose(embedded[0, first.model_phase_cos_index], np.sqrt(3) / 2)
    np.testing.assert_allclose(embedded[0, first.model_phase_sin_index], 0.5)
    np.testing.assert_allclose(phase_embedding_to_phasor(embedded, mapping), physical_row())


def test_wraparound_embeddings_keep_amplitude_and_are_close():
    mapping = representation()
    embedded = phasor_to_phase_embedding(
        np.concatenate((physical_row(100, 179), physical_row(100, -179))), mapping
    )
    first = mapping.phasors[0]
    np.testing.assert_array_equal(embedded[:, first.model_amplitude_index], [100, 100])
    phase_pair = embedded[:, [first.model_phase_cos_index, first.model_phase_sin_index]]
    assert np.linalg.norm(phase_pair[0] - phase_pair[1]) < 0.04


def test_inverse_uses_dedicated_amplitude_with_non_unit_phase_pair():
    mapping = representation()
    embedded = torch.as_tensor(phasor_to_phase_embedding(physical_row(), mapping))
    first = mapping.phasors[0]
    embedded[0, first.model_amplitude_index] = 99.5
    embedded[0, first.model_phase_cos_index] *= 7
    embedded[0, first.model_phase_sin_index] *= 7
    physical = phase_embedding_to_phasor(embedded, mapping)
    assert physical[0, first.physical_amplitude_index].item() == 99.5
    torch.testing.assert_close(physical[0, first.physical_phase_index], torch.tensor(30.0, dtype=physical.dtype))


def test_inverse_scaling_precedes_physical_conversion():
    mapping = representation()
    embedded = torch.as_tensor(phasor_to_phase_embedding(physical_row(), mapping), dtype=torch.float32)
    scaler = MinMaxScaler().fit(torch.cat((embedded, embedded + 1)))
    restored = inverse_transform_physical(scaler.transform(embedded), scaler, mapping)
    torch.testing.assert_close(restored, torch.as_tensor(physical_row(), dtype=torch.float32))


def test_old_14d_checkpoint_is_rejected_explicitly():
    checkpoint = {"scaler_state_dict": {"minimum": torch.zeros(1, 14)}}
    try:
        validate_checkpoint_model_dimension(checkpoint, 20)
    except ValueError as error:
        assert "dimension 14" in str(error)
        assert "scratch" in str(error)
    else:
        raise AssertionError("Un checkpoint 14D aurait dû être refusé")
