"""DODO bipedal robot constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

DODO_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "DODO" / "xmls" / "dodo.xml"
)
assert DODO_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(DODO_XML))


# URDF gives effort=27 Nm, velocity=6 rad/s for all 8 joints.
# Without rotor/gear specs, pick conservative defaults:
#   armature = 0.01           (small reflected inertia)
#   natural_freq = 10 Hz      (typical PD bandwidth)
#   damping_ratio = 2.0       (overdamped, matches G1 convention)
ARMATURE = 0.01
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ
EFFORT_LIMIT = 27.0

DODO_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "hip_(left|right)",
    "upper_leg_(left|right)",
    "lower_leg_(left|right)",
    "foot_(left|right)",
  ),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
)

# Standing pose. Root pos z=0.4468 matches the URDF body height.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.4468),
  joint_pos={
    "hip_.*": 0.0,
    "upper_leg_.*": 0.0,
    "lower_leg_.*": 0.0,
    "foot_.*": 0.0,
  },
  joint_vel={".*": 0.0},
)

# Slight knee bend — more stable starting pose for PPO.
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.40),
  joint_pos={
    "hip_.*": 0.0,
    "upper_leg_.*": 0.3,
    "lower_leg_.*": -0.6,
    "foot_.*": 0.3,
  },
  joint_vel={".*": 0.0},
)

# Full collisions including self-collisions. Feet get condim=3 with friction;
# everything else condim=1.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^foot_sole_(left|right)_collision$": 3, ".*_collision": 1},
  priority={r"^foot_sole_(left|right)_collision$": 1},
  friction={r"^foot_sole_(left|right)_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^foot_sole_(left|right)_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

DODO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(DODO_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_dodo_robot_cfg() -> EntityCfg:
  """Get a fresh DODO robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=DODO_ARTICULATION,
  )


DODO_ACTION_SCALE: dict[str, float] = {}
for a in DODO_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    DODO_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_dodo_robot_cfg())
  viewer.launch(robot.spec.compile())
