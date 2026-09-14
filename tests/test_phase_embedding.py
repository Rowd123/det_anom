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


def sequence_columns():
    return ["FREQ", "DFDT"] + [
        name
        for quantity in ("VA_V0", "VA_V1", "VA_V2", "IA_I0", "IA_I1", "IA_I2")
        for name in (f"VIGY_1:{quantity}_MAG", f"VIGY_1:{quantity}_ANG")
    ]


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


def test_zero_sequence_phase_exclusion_is_configurable_and_indices_are_coherent():
    full = build_feature_representation(
        sequence_columns(), exclude_zero_sequence_phase=False
    )
    reduced = build_feature_representation(
        sequence_columns(), exclude_zero_sequence_phase=True
    )

    assert full.physical_dim == 14
    assert full.model_dim == 20
    assert not full.excluded_model_names
    assert full.model_names == (
        "freq", "dfdt",
        "vigy_1:va_v0_mag", "vigy_1:va_v0_phase_cos", "vigy_1:va_v0_phase_sin",
        "vigy_1:va_v1_mag", "vigy_1:va_v1_phase_cos", "vigy_1:va_v1_phase_sin",
        "vigy_1:va_v2_mag", "vigy_1:va_v2_phase_cos", "vigy_1:va_v2_phase_sin",
        "vigy_1:ia_i0_mag", "vigy_1:ia_i0_phase_cos", "vigy_1:ia_i0_phase_sin",
        "vigy_1:ia_i1_mag", "vigy_1:ia_i1_phase_cos", "vigy_1:ia_i1_phase_sin",
        "vigy_1:ia_i2_mag", "vigy_1:ia_i2_phase_cos", "vigy_1:ia_i2_phase_sin",
    )
    assert reduced.physical_dim == 14
    assert reduced.model_dim == 16
    assert len(reduced.model_names) == reduced.model_dim
    assert set(reduced.excluded_model_names) == {
        "vigy_1:va_v0_phase_cos", "vigy_1:va_v0_phase_sin",
        "vigy_1:ia_i0_phase_cos", "vigy_1:ia_i0_phase_sin",
    }
    assert "vigy_1:va_v0_mag" in reduced.model_names
    assert "vigy_1:ia_i0_mag" in reduced.model_names
    for sequence in ("v1", "v2", "i1", "i2"):
        assert any(
            name.endswith(f"{sequence}_phase_cos") for name in reduced.model_names
        )
        assert any(
            name.endswith(f"{sequence}_phase_sin") for name in reduced.model_names
        )

    assert [feature.sequence for feature in reduced.phasors] == [0, 1, 2, 0, 1, 2]
    assert [feature.model_amplitude_index for feature in reduced.phasors] == [2, 3, 6, 9, 10, 13]
    assert [feature.model_phase_cos_index for feature in reduced.phasors] == [
        None, 4, 7, None, 11, 14,
    ]


def test_reduced_embedding_is_finite_and_inverse_uses_finite_phase_placeholder():
    mapping = build_feature_representation(
        sequence_columns(), exclude_zero_sequence_phase=True
    )
    row = physical_row(amplitude=1e-8, phase=179.0)
    embedded = phasor_to_phase_embedding(row, mapping)
    restored = phase_embedding_to_phasor(embedded, mapping)

    assert embedded.shape == (1, 16)
    assert np.isfinite(embedded).all()
    assert np.isfinite(restored).all()
    np.testing.assert_array_equal(restored[0, [3, 9]], [0.0, 0.0])
    retained_indices = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13]
    np.testing.assert_allclose(
        restored[0, retained_indices], row[0, retained_indices]
    )


def test_checkpoint_with_full_representation_is_rejected_for_reduced_model():
    checkpoint = {
        "model_x_dim": 20,
        "model_feature_names": build_feature_representation(sequence_columns()).model_names,
    }
    reduced = build_feature_representation(
        sequence_columns(), exclude_zero_sequence_phase=True
    )
    with np.testing.assert_raises(ValueError):
        validate_checkpoint_model_dimension(
            checkpoint, reduced.model_dim, reduced.model_names
        )
