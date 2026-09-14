from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "omnis"))

from model import OmniAnomaly
from units import DistributionParameterNetwork
from trainer import SigmaEpochStatistics


class IdentityHidden(torch.nn.Module):
    def forward(self, inputs):
        return inputs


def test_fixed_decoder_uses_exact_requested_sigma_and_learns_mean():
    decoder = DistributionParameterNetwork(
        IdentityHidden(), 3, 2, sigma_mode="fixed", fixed_sigma=0.05
    )
    inputs = torch.randn(4, 5, 3)
    parameters = decoder(inputs)
    assert torch.equal(parameters["scale"], torch.full_like(parameters["scale"], 0.05))
    parameters["loc"].sum().backward()
    assert decoder.mean_layer.weight.grad is not None
    assert decoder.std_layer is None


def test_learned_decoder_keeps_softplus_floor_behavior():
    decoder = DistributionParameterNetwork(
        IdentityHidden(), 3, 2, std_epsilon=0.005, sigma_mode="learned"
    )
    parameters = decoder(torch.randn(4, 5, 3))
    assert decoder.std_layer is not None
    assert (parameters["scale"] > 0.005).all()
    assert parameters["scale"].requires_grad


def _config(mode="fixed", x_dim=2):
    return SimpleNamespace(
        window_length=4, x_dim=x_dim, z_dim=2, rnn_num_hidden=3, dense_dim=3,
        use_connected_z_q=True, use_connected_z_p=True, std_epsilon=0.005,
        decoder_sigma_mode=mode, decoder_fixed_sigma=0.05,
        get_score_on_dim=False,
    )


@pytest.mark.parametrize("mode", ["fixed", "learned"])
def test_loss_diagnostics_are_finite_detached_and_preserve_decoder_gradients(mode):
    torch.manual_seed(3)
    model = OmniAnomaly(_config(mode))
    x = torch.randn(3, 4, 2)
    metrics = model.get_per_window_loss(x, n_z=2)

    for name, value in metrics.items():
        assert torch.isfinite(value).all(), name
    for name in (
        "sigma", "nll_log_sigma", "standardized_squared_error",
        "nll_by_feature", "nll_log_sigma_by_feature",
        "standardized_squared_error_by_feature", "mse_by_feature",
    ):
        assert not metrics[name].requires_grad

    reconstructed_nll = metrics["nll_by_feature"].sum(dim=-1)
    assert torch.allclose(reconstructed_nll, metrics["nll"], atol=1e-5, rtol=1e-5)
    metrics["loss"].mean().backward()
    assert model.decoder.mean_layer.weight.grad is not None
    assert torch.isfinite(model.decoder.mean_layer.weight.grad).all()


def test_training_diagnostics_accept_reduced_feature_dimension():
    names = tuple(f"feature_{index}" for index in range(16))
    model = OmniAnomaly(_config(x_dim=len(names)))
    components = model.get_per_window_loss(torch.randn(3, 4, len(names)), n_z=2)
    assert components["nll_by_feature"].shape == (3, len(names))
    assert torch.isfinite(components["loss"]).all()
    components["loss"].mean().backward()

    statistics = SigmaEpochStatistics(near_floor_threshold=0.01, feature_names=names)
    statistics.update(torch.full((2, 3, 4, 16), 0.05))
    metrics = statistics.compute()
    assert set(metrics["sigma_by_feature"]) == set(names)
    assert all(
        feature_metrics["sigma_mean"] == pytest.approx(0.05)
        for feature_metrics in metrics["sigma_by_feature"].values()
    )
