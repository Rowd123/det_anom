import sys
from pathlib import Path

import numpy as np
import torch


from omnis.utils import (  # noqa: E402
    FEATURE_REPRESENTATION_VERSION,
    MinMaxScaler,
    build_feature_representation,
    inverse_transform_physical,
    phase_embedding_to_phasor,
    phasor_to_phase_embedding,
    resolve_data_column_order,
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


def corrected_columns():
    return ["ROCOF", "FREQ"] + [
        name
        for phasor in ("I0", "I1", "I2", "V0", "V1", "V2")
        for name in (f"{phasor}.ANG", f"{phasor}.MAG")
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


def test_correct_ang_mag_semantics_and_scaling_round_trip():
    mapping = build_feature_representation(
        corrected_columns(), exclude_zero_sequence_phase=True
    )
    physical = np.array([[
        0.02, 50.0,
        170.0, 2.0,
        30.0, 1000.0,
        -20.0, 900.0,
        -175.0, 20.0,
        -45.0, 240000.0,
        60.0, 230000.0,
    ]])
    embedded = phasor_to_phase_embedding(physical, mapping)

    names = {name: index for index, name in enumerate(mapping.model_names)}
    assert embedded[0, names["i1_mag"]] == 1000.0
    np.testing.assert_allclose(embedded[0, names["i1_phase_cos"]], np.cos(np.deg2rad(30)))
    np.testing.assert_allclose(embedded[0, names["i1_phase_sin"]], np.sin(np.deg2rad(30)))
    assert embedded[0, names["v1_mag"]] == 240000.0
    np.testing.assert_allclose(embedded[0, names["v1_phase_cos"]], np.cos(np.deg2rad(-45)))
    np.testing.assert_allclose(embedded[0, names["v1_phase_sin"]], np.sin(np.deg2rad(-45)))
    assert "i0_mag" in names and "v0_mag" in names
    assert "i0_phase_cos" not in names and "v0_phase_sin" not in names

    second = physical.copy()
    second[0, [5, 11]] = [1100.0, 250000.0]
    second[0, [4, 10]] = [35.0, -40.0]
    scaler = MinMaxScaler().fit(
        phasor_to_phase_embedding(np.concatenate((physical, second)), mapping)
    )
    scaled = scaler.transform(embedded)
    unscaled = scaler.inverse_transform(scaled)
    restored = phase_embedding_to_phasor(unscaled, mapping).numpy()
    reconstructible = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
    np.testing.assert_allclose(restored[0, reconstructible], physical[0, reconstructible])
    assert scaler.minimum[0, names["i1_mag"]] == 1000.0
    assert scaler.minimum[0, names["v1_mag"]] == 240000.0
    assert abs(scaler.minimum[0, names["i1_phase_cos"]]) <= 1.0
    assert abs(scaler.minimum[0, names["v1_phase_sin"]]) <= 1.0


def test_configured_column_order_reorders_real_columns_and_rejects_missing():
    configured = corrected_columns()
    actual = [
        (
            f"VIGY_1:VA_{name}"
            if name.startswith("V")
            else f"VIGY_1:IA_{name}"
            if name.startswith("I")
            else f"VIGY_1:{name}"
        )
        for name in reversed(configured)
    ]
    assert resolve_data_column_order(actual, configured) == list(reversed(actual))
    with np.testing.assert_raises_regex(ValueError, "manquante"):
        resolve_data_column_order(actual[:-1], configured)
    with np.testing.assert_raises_regex(ValueError, "ambiguë"):
        resolve_data_column_order(actual + ["OTHER:FREQ"], configured)
    with np.testing.assert_raises_regex(ValueError, "manquante"):
        resolve_data_column_order(["NOTFREQ"], ["FREQ"])


def test_semantically_old_checkpoint_is_rejected_even_with_same_dimension():
    mapping = build_feature_representation(
        corrected_columns(), exclude_zero_sequence_phase=True
    )
    checkpoint = {
        "model_x_dim": mapping.model_dim,
        "model_feature_names": mapping.model_names,
        "feature_representation": "optional_zero_sequence_phase_v2",
    }
    with np.testing.assert_raises_regex(ValueError, "sémantique ANG/MAG"):
        validate_checkpoint_model_dimension(
            checkpoint,
            mapping.model_dim,
            mapping.model_names,
            FEATURE_REPRESENTATION_VERSION,
        )



def test_colon_suffix_generates_same_names_and_values_as_underscore():
    colon = ['vigy_1:dfdt', 'vigy_1:freq', 'vigy_1:ia_i0:ANG', 'vigy_1:ia_i0:MAG']
    underscore = [name.replace(':ANG', '_ANG').replace(':MAG', '_MAG') for name in colon]
    a = build_feature_representation(colon)
    b = build_feature_representation(underscore)
    assert a.model_names == b.model_names
    values = np.array([[1., 50., 90., 3.], [2., 49., -179., 4.]])
    np.testing.assert_array_equal(phasor_to_phase_embedding(values, a), phasor_to_phase_embedding(values, b))


def _reported_feature_names():
    return ('vigy_1:dfdt', 'vigy_1:freq') + tuple(
        f'vigy_1:{stem}_{suffix}'
        for stem in ('ia_i0', 'ia_i1', 'ia_i2', 'va_v0', 'va_v1', 'va_v2')
        for suffix in (('mag', 'phase_cos', 'phase_sin') if stem == 'va_v1'
                       else ('phase_cos', 'phase_sin'))
    )


def _feature_checkpoint(names, version=FEATURE_REPRESENTATION_VERSION):
    return dict(model_feature_names=names, model_x_dim=len(names), feature_representation=version)


def test_reported_15_feature_checkpoint_accepts_only_separator_alias_without_mutation():
    import pytest
    import re
    expected = _reported_feature_names()
    old = tuple(re.sub(r'_(mag|phase_cos|phase_sin)$', r':_\1', n) for n in expected)
    checkpoint = _feature_checkpoint(old)
    with pytest.warns(UserWarning, match='Compatibilité de nommage'):
        validate_checkpoint_model_dimension(checkpoint, 15, expected, FEATURE_REPRESENTATION_VERSION)
    assert checkpoint['model_feature_names'] == old


def test_separator_alias_does_not_accept_wrong_order_station_quantity_or_version():
    expected = _reported_feature_names()
    variants = [
        (expected[1], expected[0], *expected[2:]),
        ('other:dfdt', *expected[1:]),
        ('vigy_1:vigy_1:dfdt', *expected[1:]),
        (*expected[:-1], 'vigy_1:va_v2_mag'),
    ]
    for wrong in variants:
        with np.testing.assert_raises_regex(ValueError, 'features modèle diffèrent'):
            validate_checkpoint_model_dimension(_feature_checkpoint(wrong), 15, expected, FEATURE_REPRESENTATION_VERSION)
    with np.testing.assert_raises_regex(ValueError, 'sémantique ANG/MAG'):
        validate_checkpoint_model_dimension(_feature_checkpoint(expected, 'old'), 15, expected, FEATURE_REPRESENTATION_VERSION)


def test_separator_alias_rejects_ambiguous_features():
    names = ('vigy_1:ia_i0:_mag', 'vigy_1:ia_i0_mag')
    with np.testing.assert_raises_regex(ValueError, 'ambigus'):
        validate_checkpoint_model_dimension(_feature_checkpoint(names), 2, names)
