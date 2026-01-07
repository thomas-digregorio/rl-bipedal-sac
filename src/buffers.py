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
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray
    ):
        """Add a transition or batch of transitions to the buffer."""
        # Ensure input is batched (N, dim)
        if obs.ndim == 1:
            obs = obs[None]
            action = action[None]
            reward = np.array(reward)[None]
            next_obs = next_obs[None]
            done = np.array(done)[None]
        
        # Ensure rewards/dones are (N, 1)
        if reward.ndim == 1: reward = reward.reshape(-1, 1)
        if done.ndim == 1: done = done.reshape(-1, 1)
        
        n_samples = obs.shape[0]
        
        # Check if we need to wrap around
        if self.ptr + n_samples > self.capacity:
            # Split into two parts
            first_chunk = self.capacity - self.ptr
            self.observations[self.ptr:] = obs[:first_chunk]
            self.actions[self.ptr:] = action[:first_chunk]
            self.rewards[self.ptr:] = reward[:first_chunk]
            self.next_observations[self.ptr:] = next_obs[:first_chunk]
            self.dones[self.ptr:] = done[:first_chunk]
            
            second_chunk = n_samples - first_chunk
            self.observations[:second_chunk] = obs[first_chunk:]
            self.actions[:second_chunk] = action[first_chunk:]
            self.rewards[:second_chunk] = reward[first_chunk:]
            self.next_observations[:second_chunk] = next_obs[first_chunk:]
            self.dones[:second_chunk] = done[first_chunk:]
            
            self.ptr = second_chunk
        else:
            self.observations[self.ptr:self.ptr+n_samples] = obs
            self.actions[self.ptr:self.ptr+n_samples] = action
            self.rewards[self.ptr:self.ptr+n_samples] = reward
            self.next_observations[self.ptr:self.ptr+n_samples] = next_obs
            self.dones[self.ptr:self.ptr+n_samples] = done
            
            self.ptr = (self.ptr + n_samples) % self.capacity
            
        self.size = min(self.size + n_samples, self.capacity)

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
