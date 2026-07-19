uv run python src/mjlab/scripts/play.py \
  --task Mjlab-Velocity-Flat-DODO \
  --checkpoint-file logs/rsl_rl/dodo_velocity/2026-05-22_14-47-12_round1-standing/model_699.pt \
  --num-envs 1

## Train

	uv run python src/mjlab/scripts/train.py Mjlab-Velocity-Flat-DODO --agent.experiment-name dodo_velocity --agent.run-name phase1-stand-from-scratch --agent.max-iterations 2000


## Open plots


## Visualize on browser

uv run python src/mjlab/scripts/play.py \
  Mjlab-Velocity-Flat-DODO \
  --checkpoint-file logs/rsl_rl/dodo_velocity/2026-05-27_01-11-25_phase1-stand-from-scratch/model_0.pt \
  --num-envs 1 \
  --viewer viser

