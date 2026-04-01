# Counterpunch

A reinforcement learning agent that learns to play Punch-Out!! for NES using PPO with a CNN policy.

## Overview

Counterpunch uses [stable-retro](https://github.com/Farama-Foundation/stable-retro) to emulate the NES and [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) to train a PPO agent from raw pixel observations. Key features:

- **PPO + CnnPolicy** — processes 4 stacked grayscale frames at 84×84
- **Filtered action space** — 9 discrete moves mapped to meaningful button combos
- **Custom reward shaping** — health deltas, knockdowns, punches landed, KO bonus, and star rewards from RAM
- **Linear LR decay** and stochastic frame skip for stable training
- **Milestone videos** — automatically records gameplay at 1%, 25%, 50%, 75%, and 100% of training

## Requirements

- Python 3.10+
- A legal copy of the Punch-Out!! NES ROM
- CUDA GPU recommended for faster training

> **ROM note:** stable-retro does not include ROMs. You must supply your own legally obtained copy.

## Installation

```bash
pip install -r requirements.txt
```

Then register your ROM with stable-retro. The importer matches by SHA1 hash — if your dump doesn't match the expected hash, copy it manually and update the sha file:

```bash
# Try the importer first
python3 -m stable_retro.import roms/

# If "Imported 0 games", copy manually
RETRO_DATA=$(python3 -c "import stable_retro, os; print(os.path.join(os.path.dirname(stable_retro.__file__), 'data/stable/PunchOut-Nes-v0'))")
cp roms/punchout.nes $RETRO_DATA/rom.nes
sha1sum roms/punchout.nes | awk '{print $1}' > $RETRO_DATA/rom.sha
```

Then generate a fight save state for your ROM:

```bash
python3 scripts/make_state.py
```

> See [MEMORY_MAP.md](MEMORY_MAP.md) for a full writeup on ROM identification, RAM address verification, and state file generation.

## Project Structure

```
counterpunch/
├── config.py          # All hyperparameters and env settings
├── train.py           # PPO training script
├── evaluate.py        # Model evaluation + video recording
├── requirements.txt
├── roms/              # Place your ROM here (not committed)
├── models/            # Saved checkpoints and best model
├── logs/              # TensorBoard logs
├── videos/            # Milestone and evaluation recordings
└── envs/
    ├── __init__.py
    └── wrappers.py    # Discretizer, reward shaping, frame skip, make_env
```

## Training

```bash
# Default run (10M timesteps, 8 parallel envs)
python3 train.py

# Shorter run to test the setup
python3 train.py --timesteps 100000

# Resume from a checkpoint
python3 train.py --resume models/checkpoints/punchout_ppo_1000000_steps.zip

# Override number of envs
python3 train.py --n-envs 4
```

Checkpoints are saved to `models/checkpoints/` every 100k steps. The best model by eval reward is saved to `models/best/`.

### Milestone Videos

Training automatically records one episode at 5 points during the run and saves them to `videos/`:

| File | When |
|------|------|
| `early_<step>.mp4` | 1% of total timesteps |
| `quarter_<step>.mp4` | 25% |
| `mid_<step>.mp4` | 50% |
| `three-quarter_<step>.mp4` | 75% |
| `final_<step>.mp4` | 100% |

These let you visually compare how the agent's play style evolves across training.

## Monitoring

```bash
tensorboard --logdir logs/
```

Key metrics to watch:

| Metric | Expected trend |
|--------|---------------|
| `eval/mean_reward` | Increasing over time |
| `train/entropy_loss` | Decreasing (agent becomes less random) |
| `train/value_loss` | Should stabilize |

## Evaluation

Each training run saves models under a timestamped directory. Find yours with:

```bash
ls models/
# e.g. 20260331_223410/
```

### Record to MP4

```bash
# Record 3 episodes to videos/
python3 evaluate.py --model models/20260331_223410/best/best_model.zip --record

# Record 5 episodes
python3 evaluate.py --model models/20260331_223410/best/best_model.zip --episodes 5 --record

# Use the final model instead of the best
python3 evaluate.py --model models/20260331_223410/punchout_ppo_final.zip --record
```

### Live Render

Renders the game in a window in real time. Requires a display — on a desktop this works out of the box. On a headless server or WSL, use a virtual framebuffer:

```bash
# Desktop
python3 evaluate.py --model models/20260331_223410/best/best_model.zip --render

# WSL / headless server (requires: apt install xvfb)
xvfb-run -s "-screen 0 768x672x24" python3 evaluate.py \
    --model models/20260331_223410/best/best_model.zip --render
```

## Action Space

| ID | Action | Buttons |
|----|--------|---------|
| 0 | NOOP | — |
| 1 | Dodge left | Left |
| 2 | Dodge right | Right |
| 3 | Block / Duck | Down |
| 4 | Right body blow | A |
| 5 | Left body blow | B |
| 6 | Right head punch | Up + A |
| 7 | Left head punch | Up + B |
| 8 | Star uppercut | Start |

## Reward Shaping

| Event | Weight |
|-------|--------|
| Opponent health decrease | +1.0 per point |
| Player health decrease | −1.0 per point |
| Opponent knocked down | +5.0 |
| Player knocked down | −5.0 |
| Punch landed | +0.1 |
| Star earned | +0.5 |
| Heart lost | −0.5 per point |
| KO (opponent health → 0) | +10.0 |

All weights are in `config.py` under `RewardConfig`.

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
| `max_episode_steps` | 13500 | ~9 min of game time per episode |
| `frame_skip` | 4 | Frames per action |
| `sticky_prob` | 0.1 | Stochastic frame skip probability |
| `frame_stack` | 4 | Stacked frames for motion |
| `n_envs` | 8 | Parallel environments |

All values are in `config.py` and can be adjusted without modifying training code.
