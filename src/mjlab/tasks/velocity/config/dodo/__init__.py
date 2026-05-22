from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import dodo_flat_env_cfg, dodo_rough_env_cfg
from .rl_cfg import dodo_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-DODO",
  env_cfg=dodo_rough_env_cfg(),
  play_env_cfg=dodo_rough_env_cfg(play=True),
  rl_cfg=dodo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-DODO",
  env_cfg=dodo_flat_env_cfg(),
  play_env_cfg=dodo_flat_env_cfg(play=True),
  rl_cfg=dodo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
