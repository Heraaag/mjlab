"""PPO runner extended with model-based value expansion (MVE).

Extends MjlabOnPolicyRunner with a GRU world model that imagines H steps
beyond the real rollout horizon. The imagined λ-return replaces the single
critic bootstrap in GAE, giving better return estimates without extra env
interactions — inspired by DreamerV3 and MVE (Feinberg et al., 2018).
"""

from __future__ import annotations

import os
import time

import torch
from rsl_rl.utils import check_nan

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.world_model import WorldModel


class DreamerRunner(MjlabOnPolicyRunner):
    """On-policy PPO runner with world-model-extended value targets.

    After each real rollout the world model is trained on the collected
    transitions. Once warmed up, it unrolls H imagined steps from the final
    real observation to produce a better bootstrap for the GAE computation.
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        imagine_horizon: int = 5,
        wm_hidden_dim: int = 256,
        wm_embed_dim: int = 128,
        wm_lr: float = 1e-3,
        wm_warmup_iters: int = 100,
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)

        self.imagine_horizon = imagine_horizon
        self.wm_warmup_iters = wm_warmup_iters

        # Infer obs and action dims from environment
        obs_td = self.env.get_observations()
        obs_dim = obs_td["actor"].shape[-1]
        action_dim = self.env.num_actions

        self.world_model = WorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=wm_hidden_dim,
            embed_dim=wm_embed_dim,
            device=device,
        )
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=wm_lr
        )

    # ------------------------------------------------------------------
    # Main training loop — mirrors OnPolicyRunner.learn() but injects
    # world model training and imagined bootstrap after each rollout.
    # ------------------------------------------------------------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            start = time.time()

            # ── Real rollout ──────────────────────────────────────────
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"]["rnd_cfg"]
                        else None
                    )
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # ── Compute returns (with imagination if warmed up) ───
                use_imagination = it >= self.wm_warmup_iters
                self._compute_returns_with_imagination(obs, use_imagination)

            # ── Train world model (needs gradients, outside inference_mode) ──
            wm_loss_dict = self._update_world_model()

            # ── PPO update ────────────────────────────────────────────
            loss_dict = self.alg.update()
            loss_dict.update(wm_loss_dict)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and self.logger.log_dir is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None and self.logger.log_dir is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    # ------------------------------------------------------------------
    # World model training
    # ------------------------------------------------------------------

    def _update_world_model(self) -> dict[str, float]:
        """Train the world model on the current rollout buffer."""
        st = self.alg.storage
        T = st.num_transitions_per_env

        # Stack transitions: [T, B, dim]
        obs = st.observations["actor"]      # [T, B, obs_dim]
        actions = st.actions                # [T, B, action_dim]
        rewards = st.rewards.squeeze(-1)    # [T, B]
        dones = st.dones.squeeze(-1)        # [T, B]

        # next_obs: shift by one, use last real obs for the final step
        next_obs = torch.cat([obs[1:], obs[-1:]], dim=0)  # type: ignore[arg-type]

        loss_tensors = self.world_model.update(
            obs.float(), actions.float(), next_obs.float(), rewards.float(), dones.float()
        )

        total_loss: torch.Tensor = loss_tensors["wm_total"]  # type: ignore[assignment]
        self.wm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.wm_optimizer.step()

        return {k: v.item() for k, v in loss_tensors.items()}  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # GAE with imagined bootstrap
    # ------------------------------------------------------------------

    def _compute_returns_with_imagination(
        self, final_obs: torch.Tensor, use_imagination: bool
    ) -> None:
        """Compute GAE returns, optionally extending with imagined horizon.

        Replaces alg.compute_returns(obs). When imagination is active the
        single-step critic bootstrap V(s_T) is replaced by a λ-return
        computed over H imagined steps — reducing the critic's error from
        the GAE target.
        """
        st = self.alg.storage

        if use_imagination:
            bootstrap = self.world_model.imagine(
                start_obs=final_obs,
                actor=self.alg.actor,
                critic=self.alg.critic,
                H=self.imagine_horizon,
                gamma=self.alg.gamma,
                lam=self.alg.lam,
            ).unsqueeze(-1)  # [B, 1] to match st.values shape
        else:
            with torch.inference_mode():
                bootstrap = self.alg.critic(final_obs).detach()

        # GAE — mirrors PPO.compute_returns but with our bootstrap
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = (
                bootstrap if step == st.num_transitions_per_env - 1
                else st.values[step + 1]
            )
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = (
                st.rewards[step]
                + next_is_not_terminal * self.alg.gamma * next_values
                - st.values[step]
            )
            advantage = (
                delta + next_is_not_terminal * self.alg.gamma * self.alg.lam * advantage
            )
            st.returns[step] = advantage + st.values[step]

        st.advantages = st.returns - st.values
        if not self.alg.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (
                st.advantages.std() + 1e-8
            )

    # ------------------------------------------------------------------
    # Checkpoint — persist world model alongside PPO weights
    # ------------------------------------------------------------------

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["wm_state_dict"] = self.world_model.state_dict()
        saved_dict["wm_optimizer_state_dict"] = self.wm_optimizer.state_dict()
        torch.save(saved_dict, path)
        if self.cfg.get("upload_model"):
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        if "wm_state_dict" in loaded_dict:
            self.world_model.load_state_dict(loaded_dict["wm_state_dict"])
            self.wm_optimizer.load_state_dict(loaded_dict["wm_optimizer_state_dict"])
        return infos
