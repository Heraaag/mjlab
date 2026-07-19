"""Dodo velocity environment configurations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots import DODO_ACTION_SCALE, get_dodo_robot_cfg
from mjlab.terrains.config import flat, hf_pyramid_slope, random_rough, wave_terrain
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def dodo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Dodo rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # --- MuJoCo Contact & Solver Stability ---
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70
  cfg.scene.num_envs = 1024

  # --- Numerical Stability Policies ---
  cfg.observations["actor"].nan_policy = "warn"
  cfg.observations["critic"].nan_policy = "warn"

  cfg.terminations["nan_detection"] = TerminationTermCfg(
    func=mdp.nan_detection, time_out=True
  )

  # --- Scene & Robot Setup ---
  cfg.scene.entities = {"robot": get_dodo_robot_cfg()}

  # --- Perceptual Scanners (Scaled to DODO's Physical Size) ---
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "body"
      # Downscaled size, resolution, and distance to match Dodo's small root height
      sensor.pattern = GridPatternCfg(size=(0.8, 0.5), resolution=0.05)
      sensor.max_distance = 2.5

  site_names = ("left_foot", "right_foot")
  geom_names = ("foot_sole_left_collision", "foot_sole_right_collision")

  # Wire foot height scan to per-foot sites
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=4)

  # --- Contact Sensors ---
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^foot_sole_(left|right)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="body", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="body", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  body_ground_cfg = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(mode="geom", entity="robot", pattern=("body_collision",)),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    body_ground_cfg,
  )

  # --- Extra Fall Terminations ---
  cfg.terminations["body_ground_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": body_ground_cfg.name},
    time_out=True,
  )

  # --- Terrain: no stairs, progress from flat → rough → slopes ---
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      size=(8.0, 8.0),
      border_width=20.0,
      num_rows=10,
      num_cols=4,
      curriculum=True,
      sub_terrains={
        "flat": flat(proportion=0.4),
        "random_rough": random_rough(proportion=0.3),
        "wave_terrain": wave_terrain(proportion=0.2),
        "hf_pyramid_slope": hf_pyramid_slope(proportion=0.1, slope_range=(0.0, 0.7)),
      },
      add_lights=True,
    )

  # --- Velocity Curriculum: staged from slow to fast ---
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (0.1, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.05, 0.05)
  twist_cmd.ranges.ang_vel_z = (-0.1, 0.1)

  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0,          "lin_vel_x": (0.1, 0.5),   "lin_vel_y": (-0.05, 0.05), "ang_vel_z": (-0.1, 0.1)},
        {"step": 1500 * 24,  "lin_vel_x": (-0.3, 0.7),  "lin_vel_y": (-0.2, 0.2),   "ang_vel_z": (-0.3, 0.3)},
        {"step": 4000 * 24,  "lin_vel_x": (-0.5, 1.0),  "lin_vel_y": (-0.3, 0.3),   "ang_vel_z": (-0.4, 0.4)},
        {"step": 8000 * 24,  "lin_vel_x": (-1.0, 1.5),  "lin_vel_y": (-0.5, 0.5),   "ang_vel_z": (-0.5, 0.5)},
      ],
    },
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DODO_ACTION_SCALE

  cfg.viewer.body_name = "body"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.7  

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("body",)

  # --- Rewards Strategy ---
  cfg.rewards["terminated"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=-1.0)
  cfg.rewards["upright"].weight = 2.0
  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["soft_landing"].weight = -0.5
  cfg.rewards["knee_flex"] = RewardTermCfg(
    func=mdp.knee_flex,
    weight=1.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r"lower_leg_.*",)),
    },
  )
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # --- Dodo Joint Pose Penalties ---
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.2}
  cfg.rewards["pose"].params["std_walking"] = {
    r"hip_.*": 0.15,        
    r"upper_leg_.*": 0.4,   
    r"lower_leg_.*": 0.45,  
    r"foot_.*": 0.25,       
  }
  cfg.rewards["pose"].params["std_running"] = {
    r"hip_.*": 0.2,
    r"upper_leg_.*": 0.6,   
    r"lower_leg_.*": 0.65,
    r"foot_.*": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("body",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("body",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # --- Custom Dodo Curriculum ---
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True
    
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=terrain_levels_vel_dodo,
    params={"command_name": "twist"},
  )

  # --- Evaluation Mode Settings ---
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

    # Restore full ranges for the viser joystick GUI (slider min is 0.1).
    play_twist = cfg.commands["twist"]
    assert isinstance(play_twist, UniformVelocityCommandCfg)
    play_twist.ranges.lin_vel_x = (-1.0, 1.5)
    play_twist.ranges.lin_vel_y = (-0.5, 0.5)
    play_twist.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def dodo_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Dodo flat terrain velocity configuration."""
  cfg = dodo_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def terrain_levels_vel_dodo(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  """DODO terrain curriculum: harder to promote, easier to avoid demotion."""
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Must cross 70% of an 8m tile (5.6m) to promote -- up from 50%
  move_up = distance > terrain_generator.size[0] * 0.7

  # Only demote if distance covered is below 25% of the commanded distance
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.25
  )
  move_down *= ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

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
