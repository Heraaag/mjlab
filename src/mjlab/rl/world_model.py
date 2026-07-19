"""GRU-based world model for model-based value expansion (MVE)."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform from DreamerV3 — compresses large reward magnitudes."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(num_layers):
        layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
        layers.append(nn.ELU())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    """Recurrent world model inspired by DreamerV3.

    Maintains a GRU hidden state that summarises history. At each step:
      - encoder maps raw observations to an embedding
      - GRU evolves the hidden state given the embedding and last action
      - obs_head reconstructs the next observation (used during imagination
        so the existing PPO actor can operate unchanged)
      - reward_head predicts the immediate reward (symlog space)
      - continue_head predicts whether the episode continues (sigmoid)

    During imagination the world model unrolls H steps from the last real
    observation. The λ-return computed over those imagined steps is used as
    a better bootstrap for the real GAE, replacing the single-step critic
    estimate at the end of the rollout.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.encoder = _mlp(obs_dim, embed_dim, hidden_dim)
        self.gru = nn.GRUCell(embed_dim + action_dim, hidden_dim)
        self.obs_head = _mlp(hidden_dim, obs_dim, hidden_dim)
        self.reward_head = _mlp(hidden_dim, 1, hidden_dim)
        self.continue_head = _mlp(hidden_dim, 1, hidden_dim)

        self.to(device)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=self.device)

    def step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One world-model step.

        Args:
            obs:    [B, obs_dim]   current observation
            action: [B, action_dim] action taken
            hidden: [B, hidden_dim] current GRU hidden state

        Returns:
            hidden_next: [B, hidden_dim]
            obs_next:    [B, obs_dim]    predicted next observation
            reward:      [B]             predicted reward (symexp applied)
            continue_p:  [B]             predicted probability of not-done
        """
        embed = self.encoder(obs)
        x = torch.cat([embed, action], dim=-1)
        hidden_next = self.gru(x, hidden)

        obs_next = self.obs_head(hidden_next)
        reward = symexp(self.reward_head(hidden_next).squeeze(-1))
        continue_p = torch.sigmoid(self.continue_head(hidden_next).squeeze(-1))

        return hidden_next, obs_next, reward, continue_p

    @torch.no_grad()
    def imagine(
        self,
        start_obs: TensorDict,
        actor: nn.Module,
        critic: nn.Module,
        H: int,
        gamma: float,
        lam: float,
    ) -> torch.Tensor:
        """Unroll H imagined steps and return the λ-return as bootstrap.

        Args:
            start_obs: [B, obs_dim] final real observation after rollout
            actor:     PPO actor (takes TensorDict obs, returns actions)
            critic:    PPO critic (takes TensorDict obs, returns values)
            H:         imagination horizon
            gamma:     discount factor
            lam:       GAE lambda

        Returns:
            bootstrap: [B] imagined λ-return to use instead of V(s_T)
        """
        # start_obs is a TensorDict from the env — extract raw tensor for world model
        obs_vec = start_obs["actor"]  # [B, obs_dim]
        B = obs_vec.shape[0]
        hidden = self.init_hidden(B)

        img_rewards: list[torch.Tensor] = []
        img_continues: list[torch.Tensor] = []
        img_values: list[torch.Tensor] = []

        for _ in range(H):
            # Build a fresh TensorDict from the predicted obs vector
            obs_td = TensorDict(
                {"actor": obs_vec, "critic": obs_vec},
                batch_size=[B],
                device=self.device,
            )

            action = actor(obs_td, stochastic_output=False).detach()
            value = critic(obs_td).detach().squeeze(-1)

            hidden, obs_vec, reward, continue_p = self.step(obs_vec, action, hidden)

            img_rewards.append(reward)
            img_continues.append(continue_p)
            img_values.append(value)

        # Bootstrap from end of imagination
        obs_td_final = TensorDict(
            {"actor": obs_vec, "critic": obs_vec},
            batch_size=[B],
            device=self.device,
        )
        last_value = critic(obs_td_final).detach().squeeze(-1)

        # λ-return backwards over imagined steps
        # R_t^λ = r_t + γ * c_t * ((1-λ)*v_t + λ*R_{t+1}^λ)
        R = last_value
        for h in reversed(range(H)):
            c = img_continues[h]
            v = img_values[h]
            r = img_rewards[h]
            R = r + gamma * c * ((1 - lam) * v + lam * R)

        return R

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Train the world model on a batch of real transitions.

        Args:
            obs:      [T, B, obs_dim]
            actions:  [T, B, action_dim]
            next_obs: [T, B, obs_dim]
            rewards:  [T, B]
            dones:    [T, B]

        Returns:
            dict of scalar losses for logging
        """
        T, B = obs.shape[:2]
        hidden = self.init_hidden(B)

        obs_loss_total = torch.tensor(0.0, device=self.device)
        reward_loss_total = torch.tensor(0.0, device=self.device)
        continue_loss_total = torch.tensor(0.0, device=self.device)

        for t in range(T):
            hidden, obs_pred, reward_pred, continue_pred = self.step(
                obs[t], actions[t], hidden
            )

            obs_loss_total += nn.functional.mse_loss(obs_pred, next_obs[t])
            reward_loss_total += nn.functional.mse_loss(
                reward_pred, symlog(rewards[t])
            )
            continue_target = (1.0 - dones[t].float())
            continue_loss_total += nn.functional.binary_cross_entropy(
                continue_pred, continue_target
            )

            # Detach hidden at episode boundaries so gradients don't flow
            # across resets
            hidden = hidden * (1.0 - dones[t].float()).unsqueeze(-1)

        total_loss = obs_loss_total + reward_loss_total + continue_loss_total

        return {
            "wm_total": total_loss,
            "wm_obs": obs_loss_total,
            "wm_reward": reward_loss_total,
            "wm_continue": continue_loss_total,
        }
