"""Tests for velocity task curriculum functions."""

from unittest.mock import Mock

import torch
from conftest import get_test_device

from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel


def _make_env(terrain_levels, terrain_types, terrain_origins, root_link_pos_w):
  device = get_test_device()
  num_envs = terrain_levels.shape[0]

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.max_episode_length_s = 20.0

  env_origins = terrain_origins[terrain_levels, terrain_types].clone()

  terrain = Mock()
  terrain.terrain_levels = terrain_levels
  terrain.terrain_types = terrain_types
  terrain.terrain_origins = terrain_origins
  terrain.env_origins = env_origins
  terrain.cfg.terrain_generator.size = (8.0, 8.0)
  terrain.cfg.terrain_generator.sub_terrains = {}

  env.scene.terrain = terrain
  env.scene.env_origins = env_origins

  asset = Mock()
  asset.data.root_link_pos_w = root_link_pos_w
  env.scene.__getitem__ = Mock(return_value=asset)

  env.command_manager.get_command = Mock(
    return_value=torch.zeros(num_envs, 3, device=device)
  )

  return env


def test_terrain_levels_vel_demotes_nan_env_to_level_zero():
  device = get_test_device()
  terrain_levels = torch.tensor([5, 3, 9, 0], dtype=torch.long, device=device)
  terrain_types = torch.tensor([2, 1, 0, 3], dtype=torch.long, device=device)
  terrain_origins = torch.arange(10 * 7 * 3, dtype=torch.float, device=device).reshape(
    10, 7, 3
  )

  root_link_pos_w = terrain_origins[terrain_levels, terrain_types].clone()
  root_link_pos_w[0, :2] = float("nan")  # Env 0 has diverged.

  env = _make_env(terrain_levels, terrain_types, terrain_origins, root_link_pos_w)
  env.termination_manager.active_terms = ["nan_detection"]
  env.termination_manager.get_term = Mock(
    return_value=torch.tensor([True, False, False, False], device=device)
  )

  env_ids = torch.arange(4, device=device)
  terrain_levels_vel(env, env_ids, command_name="twist")

  terrain = env.scene.terrain
  assert terrain.terrain_levels[0] == 0
  expected_origin = terrain_origins[0, terrain_types[0]]
  torch.testing.assert_close(terrain.env_origins[0], expected_origin)

  # Other envs are untouched.
  torch.testing.assert_close(terrain.terrain_levels[1:], terrain_levels[1:])


def test_terrain_levels_vel_without_nan_detection_term():
  """Function works unchanged when nan_detection isn't a registered termination."""
  device = get_test_device()
  terrain_levels = torch.tensor([5, 3], dtype=torch.long, device=device)
  terrain_types = torch.tensor([2, 1], dtype=torch.long, device=device)
  terrain_origins = torch.arange(10 * 7 * 3, dtype=torch.float, device=device).reshape(
    10, 7, 3
  )
  root_link_pos_w = terrain_origins[terrain_levels, terrain_types].clone()

  env = _make_env(terrain_levels, terrain_types, terrain_origins, root_link_pos_w)
  env.termination_manager.active_terms = []

  env_ids = torch.arange(2, device=device)
  terrain_levels_vel(env, env_ids, command_name="twist")

  torch.testing.assert_close(env.scene.terrain.terrain_levels, terrain_levels)
