"""Checkpoint surgery: expand a flat-terrain DODO checkpoint for rough-terrain training.

The flat env has no height_scan observation. The rough env adds 160 height_scan
dims (GridPatternCfg 1.6m×1.0m, 0.1m resolution → 16×10 rays):

  Actor:  height_scan appended at the END       (36 → 196)
  Critic: height_scan inserted at position 36   (48 → 208)
          (after 'command', before 'foot_height')

All other layers are shape-identical and are copied as-is. The first-layer
weights for new dims are zero-initialised; the network will learn them from
scratch while reusing the already-trained gait.

Iteration resets to 0 for a full training budget. Optimizer state is cleared
(Adam momentum from flat terrain is not useful for rough).

Usage:
    uv run python scripts/tools/flat_to_rough_checkpoint.py \\
        --input  logs/rsl_rl/dodo_velocity/<run>/model_15000.pt \\
        --output logs/rsl_rl/dodo_velocity/<run>/model_15000_rough.pt
"""

import argparse
from pathlib import Path

import torch

HEIGHT_SCAN_DIM = 187  # GridPatternCfg(size=(1.6, 1.0), resolution=0.1) → 17×11 (endpoints included)
ACTOR_SPLIT = 36       # height_scan appended after all flat actor dims
CRITIC_SPLIT = 36      # height_scan inserted after base+joints+cmd block


def _expand_weight(weight: torch.Tensor, insert_at: int, n_new: int) -> torch.Tensor:
    """Insert n_new zero columns at insert_at in a (out_dim, in_dim) weight."""
    zeros = torch.zeros(weight.shape[0], n_new, dtype=weight.dtype)
    return torch.cat([weight[:, :insert_at], zeros, weight[:, insert_at:]], dim=1)


def _expand_stat(tensor: torch.Tensor, insert_at: int, n_new: int, fill: float) -> torch.Tensor:
    """Expand an obs_normalizer stat (1, dim) by inserting fill at insert_at."""
    filler = torch.full((1, n_new), fill, dtype=tensor.dtype)
    return torch.cat([tensor[:, :insert_at], filler, tensor[:, insert_at:]], dim=1)


def _expand_normalizer(sd: dict, insert_at: int, n_new: int) -> None:
    """In-place expand of all three normalizer buffers in a state dict."""
    sd["obs_normalizer._mean"] = _expand_stat(sd["obs_normalizer._mean"], insert_at, n_new, 0.0)
    sd["obs_normalizer._var"]  = _expand_stat(sd["obs_normalizer._var"],  insert_at, n_new, 1.0)
    sd["obs_normalizer._std"]  = _expand_stat(sd["obs_normalizer._std"],  insert_at, n_new, 1.0)


def surgery_actor(raw: dict) -> dict:
    sd = {k: v.clone() for k, v in raw.items()}
    sd["mlp.0.weight"] = _expand_weight(sd["mlp.0.weight"], ACTOR_SPLIT, HEIGHT_SCAN_DIM)
    _expand_normalizer(sd, ACTOR_SPLIT, HEIGHT_SCAN_DIM)
    return sd


def surgery_critic(raw: dict) -> dict:
    sd = {k: v.clone() for k, v in raw.items()}
    sd["mlp.0.weight"] = _expand_weight(sd["mlp.0.weight"], CRITIC_SPLIT, HEIGHT_SCAN_DIM)
    _expand_normalizer(sd, CRITIC_SPLIT, HEIGHT_SCAN_DIM)
    return sd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  required=True, type=Path, help="Source flat checkpoint (.pt)")
    parser.add_argument("--output", required=True, type=Path, help="Destination rough checkpoint (.pt)")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    print(f"Loaded  {args.input}  (iter={ckpt['iter']})")

    actor_old = ckpt["actor_state_dict"]["mlp.0.weight"].shape
    critic_old = ckpt["critic_state_dict"]["mlp.0.weight"].shape

    actor_sd  = surgery_actor(ckpt["actor_state_dict"])
    critic_sd = surgery_critic(ckpt["critic_state_dict"])

    print(f"Actor  mlp.0.weight : {actor_old}  →  {actor_sd['mlp.0.weight'].shape}")
    print(f"Critic mlp.0.weight : {critic_old}  →  {critic_sd['mlp.0.weight'].shape}")

    # Fresh Adam state: keep hyper-params (lr, betas) but discard momentum tensors.
    fresh_optimizer = {
        "state": {},
        "param_groups": ckpt["optimizer_state_dict"]["param_groups"],
    }

    new_ckpt = {
        "actor_state_dict":     actor_sd,
        "critic_state_dict":    critic_sd,
        "optimizer_state_dict": fresh_optimizer,
        "iter":  0,   # reset for a full 20k-iteration budget on rough terrain
        "infos": {},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, args.output)
    print(f"Saved   {args.output}")


if __name__ == "__main__":
    main()
