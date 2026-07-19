from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Envs whose physics diverged to NaN/Inf this step (caught by the
  # `nan_detection` termination) are sent back to the easiest terrain
  # (level 0) instead of keeping their pre-divergence level. `distance` is
  # NaN for these envs, so move_up/move_down above are both False and
  # update_env_origins would otherwise leave their level unchanged.
  if "nan_detection" in env.termination_manager.active_terms:
    nan_mask = env.termination_manager.get_term("nan_detection")[env_ids]
    if nan_mask.any():
      assert terrain.terrain_origins is not None
      reset_ids = env_ids[nan_mask]
      terrain.terrain_levels[reset_ids] = 0
      terrain.env_origins[reset_ids] = terrain.terrain_origins[
        0, terrain.terrain_types[reset_ids]
      ]

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel_adaptive(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  base_lin_vel_x: tuple[float, float],
  base_lin_vel_y: tuple[float, float],
  base_ang_vel_z: tuple[float, float],
  init_coeff_x: float = 0.0,
  init_coeff_y: float = 0.0,
  init_coeff_yaw: float = 0.0,
  target_error_x: float = 0.05,
  target_error_y: float = 0.05,
  target_error_yaw: float = 0.05,
  up_rate: float = 0.05,
  down_rate: float = 0.05,
  min_coeff: float = 0.0,
) -> dict[str, torch.Tensor]:
  """Per-env adaptive velocity curriculum with separate x/y/yaw coefficients.

  Each env has its own coeff in [min_coeff, 1] per axis. On reset the coeff
  for that env creeps up if its per-episode tracking error was below the
  target, and steps down otherwise. The ceiling ranges (base_*) are set on
  cfg.ranges; _resample_command multiplies sampled values by the per-env
  coefficients stored on cfg.
  """
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  # Set ceiling ranges — _resample_command samples from these.
  cfg.ranges.lin_vel_x = base_lin_vel_x
  cfg.ranges.lin_vel_y = base_lin_vel_y
  cfg.ranges.ang_vel_z = base_ang_vel_z

  # Initialize per-env coefficients on first call. Separate per-axis init
  # values let a resumed run seed each axis at wherever it had progressed to
  # (curriculum coeffs live as runtime attrs on cfg and are NOT checkpointed,
  # so every resume otherwise restarts all three axes from a single value).
  if not hasattr(cfg, "_coeff_x"):
    cfg._coeff_x = torch.full((env.num_envs,), init_coeff_x, device=env.device)  # type: ignore[attr-defined]
    cfg._coeff_y = torch.full((env.num_envs,), init_coeff_y, device=env.device)  # type: ignore[attr-defined]
    cfg._coeff_yaw = torch.full((env.num_envs,), init_coeff_yaw, device=env.device)  # type: ignore[attr-defined]

  if len(env_ids) > 0:
    for coeff_attr, error_key, target in [
      ("_coeff_x", "error_vel_x", target_error_x),
      ("_coeff_y", "error_vel_y", target_error_y),
      ("_coeff_yaw", "error_vel_yaw", target_error_yaw),
    ]:
      error = command_term.metrics.get(error_key)
      if error is not None:
        coeff: torch.Tensor = getattr(cfg, coeff_attr)
        below = error[env_ids] < target
        coeff[env_ids] = torch.where(
          below,
          (coeff[env_ids] + up_rate).clamp(max=4.0),
          (coeff[env_ids] - down_rate).clamp(min=min_coeff),
        )

  coeff_x: torch.Tensor = cfg._coeff_x  # type: ignore[attr-defined]
  coeff_y: torch.Tensor = cfg._coeff_y  # type: ignore[attr-defined]
  coeff_yaw: torch.Tensor = cfg._coeff_yaw  # type: ignore[attr-defined]

  return {
    "coeff_x_mean": coeff_x.mean(),
    "coeff_x_max": coeff_x.max(),
    "coeff_y_mean": coeff_y.mean(),
    "coeff_yaw_mean": coeff_yaw.mean(),
  }


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }
