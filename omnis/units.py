# units.py

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import constraints

import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution


class RNNNetwork(nn.Module):
    """
    Entrée :
        [batch_size, window_length, input_dim]

    Sortie :
        [batch_size, window_length, dense_dim]
    """

    def __init__(
        self,
        input_dim: int,
        rnn_num_hidden: int,
        rnn_cell: str = "GRU",
        hidden_dense: int = 2,
        dense_dim: int = 200,
    ):
        super().__init__()

        rnn_classes = {
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
            "Basic": nn.RNN,
        }

        if rnn_cell not in rnn_classes:
            raise ValueError(
                "rnn_cell doit être 'GRU', 'LSTM' ou 'Basic'"
            )

        self.rnn = rnn_classes[rnn_cell](
            input_size=input_dim,
            hidden_size=rnn_num_hidden,
            batch_first=True,
        )

        dense_layers = []
        current_dim = rnn_num_hidden

        for _ in range(hidden_dense):
            dense_layers.append(
                nn.Linear(current_dim, dense_dim)
            )
            current_dim = dense_dim

        self.dense_layers = nn.ModuleList(dense_layers)

    def forward(self, x: Tensor) -> Tensor:
        # Cas contenant une dimension n_samples :
        # [n_samples, batch, time, features].
        if x.ndim == 4:
            sample_size, batch_size, time_size, feature_size = x.shape

            x = x.reshape(
                sample_size * batch_size,
                time_size,
                feature_size,
            )

            outputs, _ = self.rnn(x)

            for layer in self.dense_layers:
                outputs = layer(outputs)

            return outputs.reshape(
                sample_size,
                batch_size,
                time_size,
                -1,
            )

        if x.ndim != 3:
            raise ValueError(
                "x doit avoir la forme [batch, time, features] "
                "ou [samples, batch, time, features]"
            )

        outputs, _ = self.rnn(x)

        for layer in self.dense_layers:
            outputs = layer(outputs)

        return outputs


class SoftplusStd(nn.Module):
    """
    Équivalent de :

        softplus(dense(inputs)) + epsilon
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        return F.softplus(self.linear(inputs)) + self.epsilon


class DistributionParameterNetwork(nn.Module):
    """
    Produit les paramètres loc et scale d'une loi normale.
    """

    def __init__(
        self,
        hidden_network: nn.Module,
        hidden_dim: int,
        output_dim: int,
        std_epsilon: float = 1e-5,
    ):
        super().__init__()

        self.hidden_network = hidden_network
        self.mean_layer = nn.Linear(hidden_dim, output_dim)

        self.std_layer = SoftplusStd(
            input_dim=hidden_dim,
            output_dim=output_dim,
            epsilon=std_epsilon,
        )

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        hidden = self.hidden_network(inputs)

        return {
            "loc": self.mean_layer(hidden),
            "scale": self.std_layer(hidden),
        }


class RecurrentPosteriorInputNetwork(nn.Module):
    """
    Le RNN produit input_q, utilisé ensuite par RecurrentDistribution.
    """

    def __init__(self, hidden_network: nn.Module):
        super().__init__()

        self.hidden_network = hidden_network

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        return {
            "input_q": self.hidden_network(inputs),
        }



class GaussianRandomWalk(TorchDistribution):
    """
    Marche aléatoire gaussienne :

        z_0 ~ Normal(0, initial_scale)
        z_t ~ Normal(z_{t-1}, transition_scale)

    batch_shape :
        [batch_size]

    event_shape :
        [window_length, z_dim]
    """

    arg_constraints = {}
    support = constraints.independent(constraints.real, 2)
    has_rsample = True

    def __init__(
        self,
        batch_size: int,
        window_length: int,
        z_dim: int,
        reference: Tensor,
        initial_scale: float = 1.0,
        transition_scale: float = 1.0,
        validate_args=None,
    ):
        self.batch_size = batch_size
        self.window_length = window_length
        self.z_dim = z_dim

        self.reference = reference
        self.initial_scale = initial_scale
        self.transition_scale = transition_scale

        super().__init__(
            batch_shape=torch.Size([batch_size]),
            event_shape=torch.Size([window_length, z_dim]),
            validate_args=validate_args,
        )

    def rsample(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> Tensor:
        sample_shape = torch.Size(sample_shape)

        shape = sample_shape + self.batch_shape + (self.z_dim,)

        initial_loc = self.reference.new_zeros(shape)
        initial_scale = self.reference.new_full(
            shape,
            self.initial_scale,
        )

        z_previous = dist.Normal(
            initial_loc,
            initial_scale,
        ).rsample()

        samples = [z_previous]

        for _ in range(1, self.window_length):
            transition_scale = self.reference.new_full(
                shape,
                self.transition_scale,
            )

            z_current = dist.Normal(
                z_previous,
                transition_scale,
            ).rsample()

            samples.append(z_current)
            z_previous = z_current

        return torch.stack(samples, dim=-2)

    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        # z_0
        z_initial = value[..., 0, :]

        initial_loc = torch.zeros_like(z_initial)
        initial_scale = torch.full_like(
            z_initial,
            self.initial_scale,
        )

        log_prob = dist.Normal(
            initial_loc,
            initial_scale,
        ).log_prob(z_initial).sum(dim=-1)

        # z_t | z_{t-1}
        for time_index in range(1, self.window_length):
            z_previous = value[..., time_index - 1, :]
            z_current = value[..., time_index, :]

            transition_scale = torch.full_like(
                z_previous,
                self.transition_scale,
            )

            log_prob_t = dist.Normal(
                z_previous,
                transition_scale,
            ).log_prob(z_current).sum(dim=-1)

            log_prob = log_prob + log_prob_t

        return log_prob