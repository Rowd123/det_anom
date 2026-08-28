from omnis.recurrent_distributions import RecurrentDistribution
import torch
import torch.nn as nn

class MeanMLP(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class StdMLP(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int = 128, min_std: float = 1e-5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

        self.softplus = nn.Softplus()
        self.min_std = min_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.network(x)) + self.min_std



batch_size = 32
window_length = 100
input_dim = 16
z_dim = 8
n_samples = 1024

input_q = torch.randn(
    batch_size,
    window_length,
    input_dim,
)

mean_q_mlp = MeanMLP(
    input_dim=input_dim,
    z_dim=z_dim,
)

std_q_mlp = StdMLP(
    input_dim=input_dim,
    z_dim=z_dim,
)

q = RecurrentDistribution(
    input_q=input_q,
    mean_q_mlp=mean_q_mlp,
    std_q_mlp=std_q_mlp,
    z_dim=z_dim,
)

# Équivalent conceptuel de sample(n_samples=1024).
samples = q.rsample(torch.Size([n_samples]))

print(samples.shape)
# torch.Size([1024, 32, 100, 8])

log_q = q.log_prob(samples)

print(log_q.shape)
# torch.Size([1024, 32])

probability = log_q.exp()

