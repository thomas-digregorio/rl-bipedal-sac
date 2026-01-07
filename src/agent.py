import os
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.networks import Actor, SoftQNetwork
from src.buffers import ReplayBuffer

class SAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        replay_buffer: ReplayBuffer,
        hidden_dim: int = 256,
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,        # Initial entropy coef
        autotune: bool = True,     # Whether to automatically tune entropy
        target_entropy: float = None # Target entropy (if autotune)
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.autotune = autotune
        self.replay_buffer = replay_buffer

        # --- Actor ---
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)

        # --- Critics ---
        self.qf1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.qf2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr
        )

        # --- Target Networks ---
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        # No gradients for targets
        for p in self.qf1_target.parameters(): p.requires_grad = False
        for p in self.qf2_target.parameters(): p.requires_grad = False

        # --- Entropy Tuning ---
        if self.autotune:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            
            # Log alpha is the trainable parameter
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = alpha

        # --- Optimization: Torch Compile (PyTorch 2.0+) ---
        # Fuses kernels for faster execution
        if hasattr(torch, "compile"):
            print("Compiling agent networks...")
            self.actor = torch.compile(self.actor)
            self.qf1 = torch.compile(self.qf1)
            self.qf2 = torch.compile(self.qf2)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """
        Select action for a given state.
        If evaluate=True, returns deterministic mean action.
        """
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean) # Tanh is already in sample but good to be explicit for mean
            else:
                action, _, _ = self.actor.sample(state_t)
        return action.cpu().numpy()[0]

    def update(self, batch_size: int):
        """Update Actor, Critic, and Alpha parameters."""
        if self.replay_buffer.size < batch_size:
            return {}

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # --- 1. Critic Update ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            qf1_next_target = self.qf1_target(next_state, next_action)
            qf2_next_target = self.qf2_target(next_state, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            
            # Soft Bellman Target
            # V(s') = Q(s', a') - alpha * log_pi(a'|s')
            next_q_value = reward + (1 - done) * self.gamma * (min_qf_next_target - self.alpha * next_log_prob)

        qf1_a_values = self.qf1(state, action)
        qf2_a_values = self.qf2(state, action)
        
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # --- 2. Actor Update ---
        # Frozen critics for actor update
        for p in self.qf1.parameters(): p.requires_grad = False
        for p in self.qf2.parameters(): p.requires_grad = False

        pi, log_pi, _ = self.actor.sample(state)
        qf1_pi = self.qf1(state, pi)
        qf2_pi = self.qf2(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Maximize: E[Q(s,a) - alpha * log_pi(a|s)]
        # Minimize: E[alpha * log_pi(a|s) - Q(s,a)]
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critics
        for p in self.qf1.parameters(): p.requires_grad = True
        for p in self.qf2.parameters(): p.requires_grad = True

        # --- 3. Alpha Update ---
        alpha_loss = 0.0
        if self.autotune:
            with torch.no_grad():
                _, log_pi, _ = self.actor.sample(state)
            
            # Minimize: -alpha * (log_pi + target_entropy)
            # alpha = exp(log_alpha)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()

        # --- 4. Target Updates ---
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging metrics
        return {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.autotune else 0.0,
            "alpha": self.alpha
        }

    def save_checkpoint(self, path: str):
        """Atomic Save: Save to tmp, then rename."""
        tmp_path = path + ".tmp"
        torch.save({
            'actor': self.actor.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'log_alpha': self.log_alpha if self.autotune else None,
        }, tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.qf1.load_state_dict(checkpoint['qf1'])
        self.qf2.load_state_dict(checkpoint['qf2'])
        if self.autotune and 'log_alpha' in checkpoint and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha = self.log_alpha.exp().item()
        return True
