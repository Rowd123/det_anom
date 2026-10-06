from pathlib import Path

import pytest
import yaml

from omnis.main import load_config, cli


def test_yaml_paths_have_one_origin(tmp_path, monkeypatch):
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    config = config_dir / 'run.yaml'
    config.write_text(yaml.safe_dump({'data_dir': '../data/pmu.csv', 'save_dir': 'model',
                                    'result_dir': 'result', 'restore_checkpoint': None}))
    monkeypatch.chdir(tmp_path)
    value = load_config(config)
    assert Path(value.data_dir) == tmp_path / 'data/pmu.csv'
    assert Path(value.save_dir) == config_dir / 'model'
    assert Path(value.result_dir) == config_dir / 'result'
    assert value.restore_checkpoint is None


def test_checkpoint_cannot_be_silently_ignored_in_scratch_mode(tmp_path):
    config = tmp_path / 'run.yaml'
    config.write_text('run_mode: scratch\n')
    with pytest.raises(SystemExit) as error:
        cli(['--config', str(config), '--checkpoint', 'existing.pt'])
    assert error.value.code == 2


def test_training_restores_best_checkpoint_without_external_config(tmp_path):
    from types import SimpleNamespace
    import torch
    from omnis.model import OmniAnomaly
    from omnis.trainer import Trainer
    config = SimpleNamespace(window_length=4, x_dim=2, z_dim=2, rnn_num_hidden=3,
                             dense_dim=3, use_connected_z_q=True, use_connected_z_p=True,
                             std_epsilon=.005, decoder_sigma_mode='fixed',
                             decoder_fixed_sigma=.05, get_score_on_dim=False)
    torch.manual_seed(15)
    model = OmniAnomaly(config)
    trainer = Trainer(model, max_epoch=1, batch_size=8, valid_batch_size=8,
                      train_n_z=1, valid_n_z=1, anomaly_filter_enabled=False, device='cpu')
    checkpoint = tmp_path / 'best.pt'
    trainer.fit(torch.randn(60, 2), valid_portion=.3, checkpoint_path=checkpoint)
    state = torch.load(checkpoint, weights_only=True)
    assert trainer.global_step > 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state['model_state_dict'][name])
