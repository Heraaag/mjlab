"""Dodo velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  DODO_ACTION_SCALE,
  get_dodo_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


# continous collision detection high for rough terrain
# cmax simulatenous contacts the contact sensor can track
def dodo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Dodo rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.contact_sensor_maxmatch = 100
  cfg.sim.nconmax = 30
  cfg.scene.num_envs = 256


  cfg.scene.entities = {"robot": get_dodo_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "body"

  site_names = ("left_foot", "right_foot")
  geom_names = ("foot_sole_left_collision", "foot_sole_right_collision")

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=4)
  # changed feet diam from 0.03 to 0.02 and num samples as half
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
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DODO_ACTION_SCALE

  cfg.viewer.body_name = "body"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.7  # where the arrays are depicted in height

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("body",)


  # dodo joints: hip_(left|right), upper_leg_(left|right), lower_leg_(left|right), foot_(left|right)
  # survival bonus and punishment for falling

  # cfg.rewards["joint_acc"] = RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot")})
  # cfg.rewards["joint_torques"] = RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1e-5, params={"asset_cfg": SceneEntityCfg("robot")})

  # alive: r = 1 if not terminated, else 0. Per-step survival bonus.
  #cfg.rewards["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=0.01)
  # terminated: r = 1 on non-timeout termination. One-shot fall penalty.
  cfg.rewards["terminated"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=-5.0)
  # pose: r = exp(-mean_i((q_i - q_default_i)^2 / sigma_i^2)). sigma picked by
  # cmd speed s = ||v_cmd_xy|| + |w_cmd_z|: standing if s<thr_walk, walking if
  # s<thr_run (=1.5), else running. Smaller sigma => tighter (less deviation).
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.2}

  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r"hip_.*": 0.15,        # like G1 hip_roll — tight (lateral balance)
    r"upper_leg_.*": 0.3,   # like G1 hip_pitch — loose (stride)
    r"lower_leg_.*": 0.35,  # like G1 knee — loosest (stride amplitude)
    r"foot_.*": 0.25,       # like G1 ankle_pitch — moderate (clearance)
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r"hip_.*": 0.2,
    r"upper_leg_.*": 0.5,
    r"lower_leg_.*": 0.6,
    r"foot_.*": 0.35,
  }

  # upright: r = exp(-(g_bx^2 + g_by^2) / sigma^2), sigma=sqrt(0.2)~=0.447.
  # Gravity projected into body frame; XY components measure z-axis tilt.
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("body",)
  # body_ang_vel: c = w_x^2 + w_y^2 (z excluded; yaw is for tracking).
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("body",)

  # foot_clearance: c = 1[s>0.05] * sum_f |h_f - 0.1m| * ||v_f,xy||.
  #   |v_xy| weighting auto-excludes stance feet (only swing height penalized).
  # foot_slip:      c = 1[s>0.05] * sum_f ||v_f,xy||^2 * 1[foot_f in contact].
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  # angular_momentum: c = ||L||^2, whole-body angular momentum. Suppresses flailing.
  cfg.rewards["angular_momentum"].weight = 0.0
  # air_time: r = 1[s>0.5] * sum_f 1[0.05s < t_air_f < 0.5s]. Counts feet whose
  # current airborne duration falls in the rewarded swing-duration band.
  cfg.rewards["air_time"].weight = 0.0  # dodo has only 8 dof might need rewards
  # for lifting foot off the ground periodically

  # self_collisions: c = sum over substeps of 1[max contact force > 10 N].
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
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

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def dodo_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create DODO flat terrain velocity configuration."""
  cfg = dodo_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
