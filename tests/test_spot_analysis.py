import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'omnis'))
from analyze_spot import apply_spot, score_windows, export_results
from model import OmniAnomaly
from prediction import Predictor
from utils import TrainMinMaxNormalizer, build_feature_representation, phasor_to_phase_embedding


def config(dim=2, per_dim=False):
    return SimpleNamespace(window_length=4, x_dim=dim, z_dim=2,
        rnn_num_hidden=3, dense_dim=3, use_connected_z_q=True,
        use_connected_z_p=True, std_epsilon=.005, decoder_sigma_mode='fixed',
        decoder_fixed_sigma=.05, get_score_on_dim=per_dim)


@pytest.mark.parametrize('per_dim', [False, True])
def test_scores_match_predictor_and_top_windows_keep_original_indices(per_dim):
    model = OmniAnomaly(config(per_dim=per_dim))
    values = torch.randn(25, 2)
    excludes = np.zeros(25, dtype=bool)
    excludes[8] = True
    normalizer = TrainMinMaxNormalizer().fit(values)
    torch.manual_seed(15)
    expected, _, _ = Predictor(model, n_z=2, batch_size=3, device='cpu', last_point_only=False,
                               normalizer=normalizer).get_score(values, excludes)
    if per_dim:
        expected = expected.sum(axis=-1)
    torch.manual_seed(15)
    scores, ends, retained = score_windows(model, normalizer, values, excludes,
                                           3, 2, torch.device('cpu'), retain=3)
    np.testing.assert_allclose(scores, -expected.mean(axis=-1), rtol=1e-6)
    assert not set(ends) & {8, 9, 10, 11}
    assert set(retained) == set(ends[np.argsort(-scores, kind='stable')[:3]])
    for end, (observed, reconstructed) in retained.items():
        torch.testing.assert_close(observed, values[end-3:end+1])
        assert reconstructed.shape == observed.shape


def test_score_and_reconstruction_share_the_same_latents():
    model = OmniAnomaly(config()).eval()
    values = torch.randn(2, 4, 2)
    score, _, reconstruction = model.get_score(values, n_z=1, last_point_only=False, return_reconstruction=True)
    expected = torch.distributions.Normal(reconstruction, .05).log_prob(values).sum(-1)
    torch.testing.assert_close(score, expected)


def test_spot_uses_chronology_and_pre_update_thresholds():
    calibration = np.random.default_rng(4).exponential(size=2000)
    scores = np.array([4., 0., 100., 5., 2., 30.])
    thresholds, alarms = apply_spot(calibration, scores, q=.001, level=.95)
    np.testing.assert_array_equal(alarms, scores > thresholds)
    assert alarms[2] and alarms[-1]
    assert thresholds[1] != thresholds[0]  # first non-alarm peak updated SPOT
    assert thresholds[3] == thresholds[2]  # alarm was not assimilated


@pytest.mark.parametrize('calibration', [np.ones(1000), np.arange(10.), np.array([np.nan])])
def test_invalid_calibration_is_rejected(calibration):
    with pytest.raises(ValueError):
        apply_spot(calibration, np.array([1.]))


def test_ranked_export_has_full_physical_windows_and_scores(tmp_path):
    columns = ['ROCOF', 'FREQ'] + [name for q in
        ('VA_V0', 'VA_V1', 'VA_V2', 'IA_I0', 'IA_I1', 'IA_I2')
        for name in (f'VIGY_1:{q}_ANG', f'VIGY_1:{q}_MAG')]
    rep = build_feature_representation(columns, exclude_zero_sequence_phase=True)
    physical = torch.ones(4, 14)
    embedding = phasor_to_phase_embedding(physical, rep)
    export_results(tmp_path, np.array([1., 3., 2.]), np.array([3, 4, 5]),
                   np.array([2.5, 2.5, 2.5]), np.array([False, True, False]),
                   {4: (embedding, embedding)}, rep, 4, 100)
    rows = list(csv.DictReader((tmp_path / 'ranked_windows.csv').open()))
    assert [float(r['score']) for r in rows] == [3., 2., 1.]
    assert rows[0]['source_start'] == '101'
    data = np.load(tmp_path / rows[0]['reconstruction_file'])
    assert data['observed'].shape == (4, 12)
    np.testing.assert_allclose(data['observed'], data['reconstructed'])
    assert data['score'] == 3.
    assert (tmp_path / rows[0]['figure_file']).read_text().find('<svg') >= 0


@pytest.mark.parametrize("stride", [None, 1, 2])
def test_checkpoint_to_spot_export_end_to_end(tmp_path, stride):
    import pandas as pd
    import yaml
    from analyze_spot import main
    from utils import get_data, FEATURE_REPRESENTATION_VERSION

    columns = ['DFDT', 'FREQ'] + [name for q in
        ('I0', 'I1', 'I2', 'V0', 'V1', 'V2')
        for name in (f'{q}.ANG', f'{q}.MAG')]
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(rng.normal(size=(500, 14)), columns=columns)
    frame.insert(0, 'DATE_MESURE', pd.date_range('2026-01-01', periods=500, freq='20ms'))
    frame.to_csv(tmp_path / 'data.csv', sep=';', index=False)
    c = vars(config()) | dict(dataset='PMU', data_dir='data.csv', data_column_order=columns,
         normalization_mode='train_minmax', max_train_size=400, max_test_size=None,
         train_start=0, exclude_zero_sequence_phase=True, excluded_features=[])
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(c))
    train, _, rep = get_data('PMU', data_directory=tmp_path/'data.csv',
        max_train_size=400, data_column_order=columns, return_metadata=True,
        exclude_zero_sequence_phase=True)
    c['x_dim'] = rep.model_dim
    model = OmniAnomaly(SimpleNamespace(**c))
    norm = TrainMinMaxNormalizer().fit(train[0])
    checkpoint = dict(model_state_dict=model.state_dict(),
        normalizer_state_dict=norm.checkpoint_state(), normalization_mode=norm.mode,
        model_feature_names=rep.model_names, feature_representation=FEATURE_REPRESENTATION_VERSION, model_x_dim=rep.model_dim)
    torch.save(checkpoint, tmp_path/'model.pt')
    before = (tmp_path/'model.pt').read_bytes()
    main(SimpleNamespace(config=str(tmp_path/'config.yaml'), checkpoint=str(tmp_path/'model.pt'),
         output=str(tmp_path/'out'), top=1, batch_size=32, n_z=2, q=.001, level=.90, stride=stride,
         seed=42, device='cpu'))
    rows = list(csv.DictReader((tmp_path/'out/ranked_windows.csv').open()))
    effective_stride = stride or 4
    assert len(rows) == len(range(3, 100, effective_stride))
    calibration = np.load(tmp_path/"out/calibration_scores.npz")
    np.testing.assert_array_equal(calibration["window_end"], np.arange(3, 400, effective_stride))
    scores = np.load(tmp_path/"out/chronological_scores.npz")
    np.testing.assert_array_equal(scores["window_end"], np.arange(3, 100, effective_stride))
    import json
    metadata = json.loads((tmp_path/"out/analysis.json").read_text())
    assert metadata["stride"] == effective_stride
    assert metadata["score_aggregation"] == "window_mean"
    assert sum(bool(r['figure_file']) for r in rows) == 1
    assert (tmp_path/'out/index.html').is_file()
    assert before == (tmp_path/'model.pt').read_bytes()



@pytest.mark.parametrize('stride', [1, 2, 4, 6])
def test_stride_filters_before_inference_and_keeps_grid_with_exclusions(stride):
    class CountingModel(torch.nn.Module):
        window_length = 4
        def __init__(self):
            super().__init__()
            self.processed = 0
        def get_score(self, windows, **kwargs):
            self.processed += len(windows)
            assert kwargs["last_point_only"] is False
            return -windows.sum(-1), None, windows

    model = CountingModel()
    values = torch.arange(38, dtype=torch.float32).reshape(19, 2)
    excluded = np.zeros(19, dtype=bool)
    excluded[5] = True
    normalizer = TrainMinMaxNormalizer().fit(values)
    scores, ends, retained = score_windows(model, normalizer, values, excluded,
        3, 1, torch.device('cpu'), retain=None, stride=stride)
    expected = [end for end in range(3, 19, stride) if not end-3 <= 5 <= end]
    np.testing.assert_array_equal(ends, expected)
    assert model.processed == len(expected)
    assert len(scores) == len(expected)
    for end, (observed, reconstructed) in retained.items():
        torch.testing.assert_close(observed, values[end-3:end+1])
        torch.testing.assert_close(reconstructed, observed)


@pytest.mark.parametrize('stride', [0, -1, 1.5, True])
def test_invalid_window_stride_is_rejected(stride):
    from utils import BatchSlidingWindow
    with pytest.raises(ValueError, match='stride'):
        BatchSlidingWindow(20, 4, 2, stride=stride)



@pytest.mark.parametrize('per_dim', [False, True])
def test_full_window_score_includes_interior_event_with_normal_last_point(per_dim):
    class FixedGaussianModel(torch.nn.Module):
        window_length = 4
        def get_score(self, windows, **kwargs):
            assert kwargs['last_point_only'] is False
            logp = torch.distributions.Normal(0., 1.).log_prob(windows)
            return (logp if per_dim else logp.sum(-1)), None, torch.zeros_like(windows)

    from utils import NominalNormalizer
    values = torch.zeros(8, 2)
    values[1, 0] = 4.  # event strictly inside first window; both last points zero
    scores, ends, _ = score_windows(FixedGaussianModel(),
        NominalNormalizer([0., 0.], [1., 1.]), values, None,
        2, 1, torch.device('cpu'), stride=4)
    reference = -torch.distributions.Normal(0., 1.).log_prob(values.reshape(2, 4, 2)).sum(-1).mean(-1)
    np.testing.assert_allclose(scores, reference.numpy())
    np.testing.assert_array_equal(ends, [3, 7])
    assert scores[0] - scores[1] == pytest.approx(2.)
