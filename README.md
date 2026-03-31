# Counterpunch

A reinforcement learning agent that learns to play Punch-Out!! for NES using PPO with a CNN policy.

## Overview

Counterpunch uses [stable-retro](https://github.com/Farama-Foundation/stable-retro) to emulate the NES and [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) to train a PPO agent from raw pixel observations. Key features:

- **PPO + CnnPolicy** — processes 4 stacked grayscale frames at 84x84
- **Filtered action space** — 10 discrete moves instead of 256 raw button combos
- **Custom reward shaping** — health deltas, KO bonuses, and heart-loss penalties from RAM
- **Linear LR decay** and stochastic frame skip for stable training

## Requirements

- Python 3.10+
- A legal copy of the Punch-Out!! NES ROM (`rom.nes`)
- CUDA GPU recommended for faster training

> **ROM note:** stable-retro does not include ROMs. You must supply your own legally obtained copy.

## Installation

```bash
pip install -r requirements.txt
python -m stable_retro.import /path/to/PunchOut.nes
```

## Project Structure

```
counterpunch/
├── config.py          # All hyperparameters and env settings
├── train.py           # PPO training script
├── evaluate.py        # Model evaluation + video recording
├── requirements.txt
└── envs/
    ├── __init__.py
    └── wrappers.py    # Discretizer, reward shaping, frame skip, make_env
```

## Action Space

| ID | Action | NES Buttons |
|----|--------|-------------|
| 0 | NOOP | — |
| 1 | Left jab | B |
| 2 | Right jab | A |
| 3 | Left body blow | Down + B |
| 4 | Right body blow | Down + A |
| 5 | Dodge left | Left |
| 6 | Dodge right | Right |
| 7 | Block | Up |
| 8 | Star punch | Start |
| 9 | Duck | Down |

## Reward Shaping

| Event | Weight |
|-------|--------|
| Opponent health decrease | +1.0 per point |
| Player health decrease | -1.0 per point |
| KO (opponent health → 0) | +10.0 bonus |
| Heart lost | -0.5 per heart |
| Score increase | +0.01 per point |

## Training

```bash
# Default run (10M timesteps, 8 parallel envs)
python train.py

# Custom timesteps and starting state
python train.py --timesteps 5000000 --state Match1

# Resume from checkpoint
python train.py --resume models/checkpoints/punchout_ppo_1000000_steps.zip

# Use DummyVecEnv (single process, easier to debug)
python train.py --dummy-vec --n-envs 1 --timesteps 50000
```

Checkpoints are saved to `models/checkpoints/` every 100k steps. The best model (by eval reward) is saved to `models/best/`.

## Monitoring

```bash
tensorboard --logdir logs/
```

## Evaluation

```bash
# Run 3 episodes and record MP4s to videos/
python evaluate.py --model models/best/best_model.zip --record

# Run 5 episodes, render live
python evaluate.py --model models/best/best_model.zip --episodes 5 --render

# Evaluate on a specific state
python evaluate.py --model models/best/best_model.zip --state Match1 --record
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 2.5e-4 (linear decay) | Decays to 0 over training |
| `n_steps` | 128 | Steps per env per update |
| `batch_size` | 256 | Minibatch size |
| `n_epochs` | 4 | PPO update epochs |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing |
| `clip_range` | 0.1 | PPO clipping |
| `ent_coef` | 0.01 | Entropy bonus |
| `frame_skip` | 4 | Frames per action |
| `sticky_prob` | 0.25 | Stochastic frame skip probability |
| `frame_stack` | 4 | Stacked frames for motion |
| `n_envs` | 8 | Parallel environments |

All values are defined in `config.py` and can be adjusted without modifying training code.
