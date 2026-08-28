import torch
import torch.nn as nn
from types_ import *
from base import BaseVAE


class MVAE(BaseVAE):
    def __init__(self,
                 input_dim: int,
                 time_dim: int,
                 latent_dim : int,
                 batch_size: int,
                 eps : float = 1.0,
                 **kwargs) -> None:
        super(MVAE, self).__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.eps = eps

        self.encoder1 = nn.LSTM(input_dim, 32, batch_first=True)
        self.encoder2 = nn.LSTM(32, 16, batch_first=True)


        self.fc_mu = nn.Sequential(nn.Linear(16, latent_dim),
                                nn.ReLU())

        self.fc_var = nn.Sequential(nn.Linear(16, latent_dim),
                                    nn.ReLU())


        self.decoder1 = nn.LSTM(latent_dim, 16, batch_first=True)
        self.decoder2 = nn.LSTM(16,32, batch_first=True)
        self.decoder3 = nn.LSTM(32,input_dim, batch_first=True)


    def encode(self, inputs: Tensor) -> List[Tensor]:

        result, _ = self.encoder1(inputs)
        result, _ = self.encoder2(result)
        result = result[:, -1, :]


        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:

        result = z.unsqueeze(1)
        result = result.expand(-1, self.time_dim, -1)

        result, _ = self.decoder1(result)
        result, _ = self.decoder2(result)
        result, _ = self.decoder3(result)
        return result

    def sample(self, mu: Tensor, log_var: Tensor, **kwargs) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def reparameterize(self, z_mean: Tensor, z_log_sigma: Tensor) -> Tensor:
        epsilon = torch.randn_like(z_log_sigma) * self.eps
        return z_mean + torch.exp(z_log_sigma) * epsilon


    def forward(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        inputs = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = torch.mean(torch.log(torch.cosh(recons - inputs + 1e-12)))

        kld_loss = -0.5 *  torch.sum(1 + log_var - mu**2 - log_var.exp())

        loss = recons_loss + kld_loss

        return {'loss': loss, 'reconstruction_loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def generate(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.forward(inputs)[0]









        



