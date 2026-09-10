# recurrent_distributions.py

from typing import Callable, Optional

import torch
from torch import Tensor
from torch.distributions import constraints

import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution


class RecurrentDistribution(TorchDistribution):
    """
    Distribution gaussienne récurrente :

        z_t ~ Normal(
            mean_q_mlp([x_t, z_{t-1}]),
            std_q_mlp([x_t, z_{t-1}])
        )

    Paramètres
    ----------
    input_q:
        Tensor de forme [batch_size, window_length, input_dim].

    mean_q_mlp:
        Module/callable transformant [..., input_dim + z_dim]
        en [..., z_dim].

    std_q_mlp:
        Module/callable transformant [..., input_dim + z_dim]
        en [..., z_dim]. Sa sortie doit être strictement positive.

    z_dim:
        Dimension latente.

    window_length:
        Si fourni, doit correspondre à input_q.shape[1].
    """

    arg_constraints = {}
    support = constraints.independent(constraints.real, 2)
    has_rsample = True

    def __init__(
        self,
        input_q: Tensor,
        mean_q_mlp: Callable[[Tensor], Tensor],
        std_q_mlp: Callable[[Tensor], Tensor],
        z_dim: int,
        window_length: Optional[int] = None,
        check_numerics: bool = True,
        validate_args: Optional[bool] = None,
    ):
        if input_q.ndim != 3:
            raise ValueError(
                "input_q doit avoir la forme "
                "[batch_size, window_length, input_dim]"
            )

        batch_size, inferred_window_length, _ = input_q.shape

        if window_length is None:
            window_length = inferred_window_length
        elif window_length != inferred_window_length:
            raise ValueError(
                f"window_length={window_length}, mais input_q contient "
                f"{inferred_window_length} pas de temps"
            )

        self.input_q = input_q
        self.mean_q_mlp = mean_q_mlp
        self.std_q_mlp = std_q_mlp
        self.z_dim = z_dim
        self.window_length = window_length
        self.check_numerics = check_numerics

        # Une séquence complète [window_length, z_dim] constitue
        # un événement. Les éléments du batch sont indépendants.
        batch_shape = torch.Size([batch_size])
        event_shape = torch.Size([window_length, z_dim])

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _parameters(
        self,
        input_t: Tensor,
        z_previous: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Calcule les paramètres de q(z_t | z_{t-1}, x_t).

        input_t:
            [batch_size, input_dim]

        z_previous:
            [..., batch_size, z_dim]
        """
        # Ajoute les dimensions d'échantillonnage devant input_t.
        sample_ndims = z_previous.ndim - 2

        expanded_input_t = input_t.reshape(
            (1,) * sample_ndims + input_t.shape
        ).expand(z_previous.shape[:-1] + (input_t.shape[-1],))

        recurrent_input = torch.cat(
            [expanded_input_t, z_previous],
            dim=-1,
        )

        mean = self.mean_q_mlp(recurrent_input)
        std = self.std_q_mlp(recurrent_input)

        if mean.shape[-1] != self.z_dim:
            raise ValueError(
                "mean_q_mlp doit produire une dernière dimension "
                f"égale à z_dim={self.z_dim}, reçu {mean.shape[-1]}"
            )

        if std.shape != mean.shape:
            raise ValueError(
                "std_q_mlp et mean_q_mlp doivent produire "
                f"la même forme, reçu {std.shape} et {mean.shape}"
            )

        if self.check_numerics:
            if not torch.isfinite(mean).all():
                raise ValueError("La moyenne contient des NaN ou des infinis")

            if not torch.isfinite(std).all():
                raise ValueError(
                    "L'écart-type contient des NaN ou des infinis"
                )

            if not (std > 0).all():
                raise ValueError(
                    "std_q_mlp doit produire des valeurs strictement positives"
                )

        return mean, std

    def rsample(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> Tensor:
        """
        Retourne un tenseur de forme :

            sample_shape + [batch_size, window_length, z_dim]

        Par exemple, rsample([1024]) retourne :

            [1024, batch_size, window_length, z_dim]
        """
        sample_shape = torch.Size(sample_shape)

        z_previous = self.input_q.new_zeros(
            sample_shape + self.batch_shape + (self.z_dim,)
        )

        samples = []

        for time_index in range(self.window_length):
            input_t = self.input_q[:, time_index, :]

            mean_t, std_t = self._parameters(
                input_t=input_t,
                z_previous=z_previous,
            )

            # rsample() applique la reparamétrisation :
            # z_t = mean_t + std_t * epsilon
            z_t = dist.Normal(mean_t, std_t).rsample()

            samples.append(z_t)
            z_previous = z_t

        # Temps placé juste avant z_dim :
        # [..., batch_size, window_length, z_dim]
        return torch.stack(samples, dim=-2)

    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> Tensor:
        # Contrairement à rsample(), sample() ne construit pas
        # de graphe de gradient.
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Calcule la log-probabilité de séquences de forme :

            [..., batch_size, window_length, z_dim]

        Le résultat est de forme :

            [..., batch_size]

        Les dimensions window_length et z_dim sont additionnées,
        car elles appartiennent à event_shape.
        """
        expected_suffix = self.batch_shape + self.event_shape

        if value.shape[-len(expected_suffix):] != expected_suffix:
            raise ValueError(
                "La valeur doit se terminer par la forme "
                f"{tuple(expected_suffix)}, reçu {tuple(value.shape)}"
            )

        z_previous = value.new_zeros(
            value.shape[:-2] + (self.z_dim,)
        )

        log_probs = []

        for time_index in range(self.window_length):
            input_t = self.input_q[:, time_index, :]
            z_t = value[..., time_index, :]

            mean_t, std_t = self._parameters(
                input_t=input_t,
                z_previous=z_previous,
            )

            # Forme : [..., batch_size, z_dim]
            log_prob_t = dist.Normal(mean_t, std_t).log_prob(z_t)

            # Somme sur z_dim.
            log_probs.append(log_prob_t.sum(dim=-1))

            # La distribution au temps suivant dépend de z_t.
            z_previous = z_t

        # Somme sur le temps.
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def scale_statistics(self, value: Tensor) -> Tensor:
        """Retourne tous les écarts-types conditionnels évalués sur ``value``."""
        z_previous = value.new_zeros(value.shape[:-2] + (self.z_dim,))
        scales = []
        for time_index in range(self.window_length):
            _, scale_t = self._parameters(
                self.input_q[:, time_index, :],
                z_previous,
            )
            scales.append(scale_t)
            z_previous = value[..., time_index, :]
        return torch.stack(scales, dim=-2)
