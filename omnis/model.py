# model.py

import torch
from torch import Tensor, nn

import pyro.distributions as dist

from units import GaussianRandomWalk, DistributionParameterNetwork
from units import RecurrentPosteriorInputNetwork, RNNNetwork, SoftplusStd
from recurrent_distributions import RecurrentDistribution
from vae import VAE


make_normal = lambda loc, scale: dist.Normal(loc, scale)

class RecurrentPosteriorFactory(nn.Module):
    """
    Construit RecurrentDistribution à partir d'input_q.
    """

    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        window_length: int,
        std_epsilon: float,
    ):
        super().__init__()

        self.mean_layer = nn.Linear(
            input_dim + z_dim,
            z_dim,
        )

        self.std_layer = SoftplusStd(
            input_dim=input_dim + z_dim,
            output_dim=z_dim,
            epsilon=std_epsilon,
        )

        self.z_dim = z_dim
        self.window_length = window_length

    def forward(self, input_q: Tensor):
        return RecurrentDistribution(
            input_q=input_q,
            mean_q_mlp=self.mean_layer,
            std_q_mlp=self.std_layer,
            z_dim=self.z_dim,
            window_length=self.window_length,
        )


class OmniAnomaly(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.window_length = config.window_length
        self.x_dims = config.x_dim
        self.z_dims = config.z_dim

        # ------------------------------------------------------------
        # Encodeur q(z | x)
        # ------------------------------------------------------------

        encoder_rnn = RNNNetwork(
            input_dim=config.x_dim,
            rnn_num_hidden=config.rnn_num_hidden,
            rnn_cell="GRU",
            hidden_dense=2,
            dense_dim=config.dense_dim,
        )

        if config.use_connected_z_q:
            self.encoder = RecurrentPosteriorInputNetwork(
                encoder_rnn
            )

            self.posterior_factory = RecurrentPosteriorFactory(
                input_dim=config.dense_dim,
                z_dim=config.z_dim,
                window_length=config.window_length,
                std_epsilon=config.std_epsilon,
            )

            q_z_given_x = self.posterior_factory

            # RecurrentDistribution possède déjà :
            # event_shape = [window_length, z_dim].
            z_event_ndims = 0

        else:
            self.encoder = DistributionParameterNetwork(
                hidden_network=encoder_rnn,
                hidden_dim=config.dense_dim,
                output_dim=config.z_dim,
                std_epsilon=config.std_epsilon,
            )

            q_z_given_x = make_normal

            # Normal produit initialement :
            # batch_shape = [batch, time, z_dim].
            # Les deux dernières dimensions deviennent l'événement.
            z_event_ndims = 2

        # ------------------------------------------------------------
        # Décodeur p(x | z)
        # ------------------------------------------------------------

        decoder_rnn = RNNNetwork(
            input_dim=config.z_dim,
            rnn_num_hidden=config.rnn_num_hidden,
            rnn_cell="GRU",
            hidden_dense=2,
            dense_dim=config.dense_dim,
        )

        self.decoder = DistributionParameterNetwork(
            hidden_network=decoder_rnn,
            hidden_dim=config.dense_dim,
            output_dim=config.x_dim,
            std_epsilon=config.std_epsilon,
        )

        # ------------------------------------------------------------
        # Prior p(z)
        # ------------------------------------------------------------

        if config.use_connected_z_p:
            def make_prior(reference: Tensor):
                return GaussianRandomWalk(
                    batch_size=reference.shape[0],
                    window_length=config.window_length,
                    z_dim=config.z_dim,
                    reference=reference,
                )

            # GaussianRandomWalk possède déjà son event_shape.
            prior_event_ndims = 0

        else:
            def make_prior(reference: Tensor):
                return dist.Normal(
                    reference.new_zeros(
                        reference.shape[0],
                        config.window_length,
                        config.z_dim,
                    ),
                    reference.new_ones(
                        reference.shape[0],
                        config.window_length,
                        config.z_dim,
                    ),
                )

            prior_event_ndims = 2

        if prior_event_ndims != z_event_ndims:
            raise ValueError(
                "Le VAE générique doit gérer séparément les event_ndims "
                "du prior et du posterior"
            )

        self.vae = VAE(
            p_z=make_prior,
            p_x_given_z=make_normal,
            q_z_given_x=q_z_given_x,
            h_for_p_x=self.decoder,
            h_for_q_z=self.encoder,
            z_event_ndims=z_event_ndims,
            x_event_ndims=2,
        )

    def forward(
        self,
        x: Tensor,
        n_z: int | None = None,
    ) -> Tensor:
        return self.vae(
            x,
            n_z=n_z,
        )

    def get_training_loss(
        self,
        x: Tensor,
        n_z: int = 1,
    ) -> Tensor:
        return self.vae.get_training_loss(
            x=x,
            n_z=n_z,
            objective="elbo",
        )

    def get_per_window_loss(self, x: Tensor, n_z: int = 1) -> dict[str, Tensor]:
        return self.vae.get_per_window_loss(x=x, n_z=n_z)

    @torch.no_grad()
    def get_score(
        self,
        x: Tensor,
        n_z: int = 1,
        last_point_only: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Retourne :
            reconstruction_log_prob, z_statistics
        """
        posterior = self.vae.posterior(x)

        z_samples = posterior.rsample(
            torch.Size([n_z])
        )

        # statistiques des latents.
        z_mean = z_samples.mean(dim=0)

        if n_z > 1:
            z_std = z_samples.std(
                dim=0,
                unbiased=True,
            )
        else:
            z_std = torch.zeros_like(z_mean)

        z_statistics = torch.cat(
            [z_mean, z_std],
            dim=-1,
        )

        # Paramètres de p(x | z).
        x_parameters = self.decoder(z_samples)

        reconstruction_distribution = dist.Normal(
            x_parameters["loc"],
            x_parameters["scale"],
        )

        # [n_z, batch, time, x_dim]
        reconstruction_log_prob = (
            reconstruction_distribution.log_prob(x)
        )

        # Moyenne sur les échantillons z.
        reconstruction_log_prob = (
            reconstruction_log_prob.mean(dim=0)
        )

        if not self.config.get_score_on_dim:
            # Somme sur x_dim :
            # [batch, time, x_dim] -> [batch, time]
            reconstruction_log_prob = (
                reconstruction_log_prob.sum(dim=-1)
            )

        if last_point_only:
            # Dernier instant de chaque fenêtre.
            reconstruction_log_prob = (
                reconstruction_log_prob[:, -1]
            )

        return reconstruction_log_prob, z_statistics
