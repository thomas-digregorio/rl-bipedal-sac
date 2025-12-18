import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

def weights_init_(m):
    """Custom weight init for better convergence."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class SoftQNetwork(nn.Module):
    """Critic Network: Q(s, a)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class Actor(nn.Module):
    """Actor Network: pi(a|s). Gaussian Policy."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared features
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self.apply(weights_init_)

    def forward(self, state):
        x = self.trunk(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        
        # Clamp log_std to maintain numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        # Reparameterization trick (rsample)
        x_t = normal.rsample()  
        
        # Squash to [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Enforce action bounds
        action = y_t
        
        # Calculate log_prob
        # log_prob(tanh(x)) = log_prob(x) - sum(log(1 - tanh(x)^2))
        log_prob = normal.log_prob(x_t)
        
        # Correction for Tanh squashing
        # epsilon for numerical stability
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean
