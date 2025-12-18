import numpy as np
import torch

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device

        # Preallocate memory
        # Using float32 for standard Gym environments
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add a transition to the buffer."""
        # Handle simple adds
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.as_tensor(self.observations[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device),
            torch.as_tensor(self.rewards[idxs], device=self.device),
            torch.as_tensor(self.next_observations[idxs], device=self.device),
            torch.as_tensor(self.dones[idxs], device=self.device),
        )
