# vae.py
from math import log, pi
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

import pyro
from pyro.infer import RenyiELBO, SVI, Trace_ELBO


DistributionFactory = Callable[..., Distribution]
PriorFactory = Callable[[Tensor], Distribution]


class VAE(nn.Module):
    """
    VAE générique écrit avec PyTorch et Pyro.

    Le modèle génératif est :

        z ~ p(z)
        x ~ p(x | z)

    Le guide variationnel est :

        z ~ q(z | x)

    Paramètres
    ----------
    p_z:
        Fonction recevant un tenseur de référence de forme
        [batch_size, ...] et retournant la distribution a priori p(z).

    p_x_given_z:
        Fonction construisant p(x | z) à partir des paramètres produits
        par h_for_p_x.

    q_z_given_x:
        Fonction construisant q(z | x) à partir des paramètres produits
        par h_for_q_z.

    h_for_p_x:
        Décodeur. Il reçoit z et retourne un dictionnaire contenant les
        paramètres de p(x | z).

    h_for_q_z:
        Encodeur. Il reçoit x et retourne un dictionnaire contenant les
        paramètres de q(z | x).

    z_event_ndims:
        Nombre de dernières dimensions appartenant à l'événement z.

    x_event_ndims:
        Nombre de dernières dimensions appartenant à l'événement x.
    """

    def __init__(
        self,
        p_z: PriorFactory,
        p_x_given_z: DistributionFactory,
        q_z_given_x: DistributionFactory,
        h_for_p_x: nn.Module,
        h_for_q_z: nn.Module,
        z_event_ndims: int = 1,
        x_event_ndims: int = 1,
    ):
        super().__init__()

        if not callable(p_z):
            raise TypeError("p_z doit être appelable")

        if not callable(p_x_given_z):
            raise TypeError("p_x_given_z doit être appelable")

        if not callable(q_z_given_x):
            raise TypeError("q_z_given_x doit être appelable")

        if not callable(h_for_p_x):
            raise TypeError("h_for_p_x doit être appelable")

        if not callable(h_for_q_z):
            raise TypeError("h_for_q_z doit être appelable")

        if z_event_ndims < 0:
            raise ValueError("z_event_ndims doit être positif ou nul")

        if x_event_ndims < 0:
            raise ValueError("x_event_ndims doit être positif ou nul")

        self.p_z = p_z
        self.p_x_given_z = p_x_given_z
        self.q_z_given_x = q_z_given_x

        self.h_for_p_x = h_for_p_x
        self.h_for_q_z = h_for_q_z

        self.z_event_ndims = z_event_ndims
        self.x_event_ndims = x_event_ndims

        # Ce buffer permet de connaître le device et le dtype du modèle,
        # notamment lorsqu'on génère des données sans x.
        self.register_buffer("_reference", torch.tensor(0.0))

    @staticmethod
    def _to_event(
        distribution: Distribution,
        event_ndims: int,
    ) -> Distribution:
        """
        Transforme les dernières dimensions de batch en dimensions
        d'événement.

        """
        if event_ndims == 0:
            return distribution

        return distribution.to_event(event_ndims)

    def prior(self, reference: Tensor) -> Distribution:
        """
        Construit la distribution a priori p(z).

        reference sert principalement à déterminer :
        - le batch_size ;
        - le device ;
        - le dtype.
        """
        prior_distribution = self.p_z(reference)

        if not isinstance(prior_distribution, Distribution):
            raise TypeError("p_z doit retourner une Distribution")

        return self._to_event(
            prior_distribution,
            self.z_event_ndims,
        )

    def posterior(self, x: Tensor) -> Distribution:
        """
        Construit q(z | x).
        """
        z_parameters = self.h_for_q_z(x)

        if not isinstance(z_parameters, dict):
            raise TypeError(
                "h_for_q_z doit retourner un dictionnaire de paramètres"
            )

        posterior_distribution = self.q_z_given_x(**z_parameters)

        if not isinstance(posterior_distribution, Distribution):
            raise TypeError(
                "q_z_given_x doit retourner une Distribution"
            )

        return self._to_event(
            posterior_distribution,
            self.z_event_ndims,
        )

    def likelihood(self, z: Tensor) -> Distribution:
        """
        Construit p(x | z).
        """
        x_parameters = self.h_for_p_x(z)

        if not isinstance(x_parameters, dict):
            raise TypeError(
                "h_for_p_x doit retourner un dictionnaire de paramètres"
            )

        likelihood_distribution = self.p_x_given_z(**x_parameters)

        if not isinstance(likelihood_distribution, Distribution):
            raise TypeError(
                "p_x_given_z doit retourner une Distribution"
            )

        return self._to_event(
            likelihood_distribution,
            self.x_event_ndims,
        )

    def model(
        self,
        x: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        z: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Modèle génératif :

            z ~ p(z)
            x ~ p(x | z)

        Si x est fourni, il est traité comme une observation.
        Sinon, une nouvelle valeur x est générée.
        """
        pyro.module("vae", self)

        if x is not None:
            batch_size = x.shape[0]
            reference = x
        else:
            if batch_size is None:
                raise ValueError(
                    "batch_size est obligatoire lorsque x n'est pas fourni"
                )

            reference = self._reference.new_zeros(batch_size, 1)

        p_z = self.prior(reference)

        with pyro.plate("data", batch_size, dim=-1):
            # Si z est fourni, Pyro le traite comme une valeur observée.
            z_value = pyro.sample(
                "z",
                p_z,
                obs=z,
            )

            p_x_given_z = self.likelihood(z_value)

            x_value = pyro.sample(
                "x",
                p_x_given_z,
                obs=x,
            )

        return x_value

    def guide(
        self,
        x: Tensor,
        batch_size: Optional[int] = None,
        z: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Guide variationnel :

            z ~ q(z | x)

        Il approxime le posterior réel p(z | x).
        """
        del batch_size

        pyro.module("vae", self)

        q_z_given_x = self.posterior(x)

        with pyro.plate("data", x.shape[0], dim=-1):
            z_value = pyro.sample(
                "z",
                q_z_given_x,
                obs=z,
            )

        return z_value

    def forward(
        self,
        x: Tensor,
        n_z: Optional[int] = None,
    ) -> Tensor:
        """
        Encode x en échantillonnant z ~ q(z | x).

        Sans n_z :
            résultat [batch_size, ..., z_dim]

        Avec n_z :
            résultat [n_z, batch_size, ..., z_dim]
        """
        q_z_given_x = self.posterior(x)

        if n_z is None:
            return q_z_given_x.rsample()

        if n_z < 1:
            raise ValueError("n_z doit être supérieur ou égal à 1")

        return q_z_given_x.rsample(torch.Size([n_z]))

    @torch.no_grad()
    def reconstruct(
        self,
        x: Tensor,
        n_z: Optional[int] = None,
        n_x: Optional[int] = None,
        use_mean: bool = False,
    ) -> Tensor:
        """
        Reconstruit x en suivant :

            x -> q(z | x) -> z -> p(x | z) -> x_reconstructed
        """
        q_z_given_x = self.posterior(x)

        if n_z is None:
            z = q_z_given_x.rsample()
        else:
            if n_z < 1:
                raise ValueError("n_z doit être supérieur ou égal à 1")

            z = q_z_given_x.rsample(torch.Size([n_z]))

        p_x_given_z = self.likelihood(z)

        if use_mean:
            return p_x_given_z.mean

        if n_x is None:
            return p_x_given_z.sample()

        if n_x < 1:
            raise ValueError("n_x doit être supérieur ou égal à 1")

        return p_x_given_z.sample(torch.Size([n_x]))

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        n_x: Optional[int] = None,
        use_mean: bool = False,
    ) -> Tensor:
        """
        Génère de nouvelles données à partir du prior :

            z ~ p(z)
            x ~ p(x | z)
        """
        if batch_size < 1:
            raise ValueError("batch_size doit être supérieur ou égal à 1")

        reference = self._reference.new_zeros(batch_size, 1)

        p_z = self.prior(reference)
        z = p_z.sample()

        p_x_given_z = self.likelihood(z)

        if use_mean:
            return p_x_given_z.mean

        if n_x is None:
            return p_x_given_z.sample()

        if n_x < 1:
            raise ValueError("n_x doit être supérieur ou égal à 1")

        return p_x_given_z.sample(torch.Size([n_x]))

    def get_training_loss(
        self,
        x: Tensor,
        n_z: int = 1,
        objective: str = "elbo",
    ) -> Tensor:
        """
        Calcule une perte différentiable.

        objective="elbo" :
            ELBO Monte-Carlo standard.

        objective="iwae" :
            Objectif IWAE, via RenyiELBO avec alpha=0.
        """
        if n_z < 1:
            raise ValueError("n_z doit être supérieur ou égal à 1")

        if objective == "elbo":
            elbo = Trace_ELBO(
                num_particles=n_z,
                vectorize_particles=n_z > 1,
            )
        elif objective == "iwae":
            elbo = RenyiELBO(
                alpha=0.0,
                num_particles=n_z,
                vectorize_particles=n_z > 1,
            )
        else:
            raise ValueError(
                "objective doit être égal à 'elbo' ou 'iwae'"
            )

        return elbo.differentiable_loss(
            self.model,
            self.guide,
            x,
        )

    def get_per_window_loss(self, x: Tensor, n_z: int = 1) -> dict[str, Tensor]:
        """Calcule la même ELBO Monte-Carlo, sans réduire le batch Pyro."""
        if n_z < 1:
            raise ValueError("n_z doit être supérieur ou égal à 1")

        posterior = self.posterior(x)
        z = posterior.rsample(torch.Size([n_z]))
        prior = self.prior(x)
        likelihood = self.likelihood(z)

        # _to_event encapsule la Normal dans une Independent. Son base_dist
        # reste la Normal construite par le décodeur et porte son scale.
        likelihood_base = likelihood
        while not hasattr(likelihood_base, "scale"):
            if not hasattr(likelihood_base, "base_dist"):
                raise TypeError(
                    "p(x | z) doit exposer le scale de sa loi normale"
                )
            likelihood_base = likelihood_base.base_dist
        # Pour Normal(mu, sigma), par cellule :
        # -log p(x|z) = log(sigma) + 0.5*((x-mu)/sigma)^2 + 0.5*log(2*pi).
        # Ces vues par feature sont moyennées sur MC, puis sommées sur le temps.
        # no_grad garantit que l'instrumentation ne prolonge pas le graphe.
        with torch.no_grad():
            sigma = likelihood_base.scale.detach()
            loc = likelihood_base.loc.detach()
            standardized_squared_error = ((x - loc) / sigma).square()
            nll_log_sigma_by_feature = torch.log(sigma).mean(dim=0).sum(dim=1)
            standardized_squared_error_by_feature = (
                standardized_squared_error.mean(dim=0).sum(dim=1)
            )
            nll_by_feature = (
                nll_log_sigma_by_feature
                + 0.5 * standardized_squared_error_by_feature
                + x.shape[-2] * 0.5 * log(2.0 * pi)
            )
            mse_by_feature = (loc - x).square().mean(dim=0).mean(dim=1)

        log_px = likelihood.log_prob(x)
        log_pz = prior.log_prob(z)
        log_qz = posterior.log_prob(z)
        nll = -log_px.mean(dim=0)
        kl = (log_qz - log_pz).mean(dim=0)
        per_window_loss = nll + kl
        reconstruction_mean = likelihood.mean.mean(dim=0)
        mse = (reconstruction_mean - x).square().mean(dim=(-2, -1))

        return {
            "loss": per_window_loss,
            "nll": nll,
            "kl": kl,
            "mse": mse,
            # Vue détachée du scale de likelihood : le Trainer peut ainsi
            # instrumenter sigma sans conserver ni modifier le graphe de loss.
            "sigma": sigma,
            "nll_log_sigma": nll_log_sigma_by_feature.sum(dim=-1).detach(),
            "standardized_squared_error": (
                standardized_squared_error_by_feature.sum(dim=-1).detach()
            ),
            "nll_by_feature": nll_by_feature.detach(),
            "nll_log_sigma_by_feature": nll_log_sigma_by_feature.detach(),
            "standardized_squared_error_by_feature": (
                standardized_squared_error_by_feature.detach()
            ),
            "mse_by_feature": mse_by_feature.detach(),
        }

    def make_svi(
        self,
        learning_rate: float = 1e-3,
        n_z: int = 1,
        objective: str = "elbo",
    ) -> SVI:
        """
        Construit l'objet SVI utilisé pour entraîner le VAE.
        """
        if objective == "elbo":
            loss = Trace_ELBO(
                num_particles=n_z,
                vectorize_particles=n_z > 1,
            )
        elif objective == "iwae":
            loss = RenyiELBO(
                alpha=0.0,
                num_particles=n_z,
                vectorize_particles=n_z > 1,
            )
        else:
            raise ValueError(
                "objective doit être égal à 'elbo' ou 'iwae'"
            )

        optimizer = pyro.optim.Adam({
            "lr": learning_rate,
        })

        return SVI(
            model=self.model,
            guide=self.guide,
            optim=optimizer,
            loss=loss,
        )


class Encoder(nn.Module):
    """
    Construit les paramètres de q(z | x).
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, z_dim)
        self.log_scale_layer = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        hidden = self.shared(x)

        mean = self.mean_layer(hidden)

        # clamp évite des valeurs exponentielles extrêmes.
        log_scale = self.log_scale_layer(hidden).clamp(
            min=-10.0,
            max=10.0,
        )

        scale = torch.exp(log_scale)

        return {
            "loc": mean,
            "scale": scale,
        }


class Decoder(nn.Module):
    """
    Construit les paramètres de p(x | z).
    """

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, x_dim)
        self.log_scale_layer = nn.Linear(hidden_dim, x_dim)

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        hidden = self.shared(z)

        mean = self.mean_layer(hidden)

        log_scale = self.log_scale_layer(hidden).clamp(
            min=-10.0,
            max=10.0,
        )

        scale = torch.exp(log_scale)

        return {
            "loc": mean,
            "scale": scale,
        }
