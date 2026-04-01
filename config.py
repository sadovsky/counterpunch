"""Centralized configuration for Counterpunch training."""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    game: str = "PunchOut-Nes-v0"
    state: str = "Match1"
    frame_skip: int = 4
    sticky_prob: float = 0.25
    grayscale: bool = True
    resize: tuple[int, int] = (84, 84)
    frame_stack: int = 4
    n_envs: int = 16


@dataclass
class RewardConfig:
    opponent_damage: float = 1.0
    player_damage: float = -1.0
    ko_bonus: float = 10.0
    knockdown_dealt: float = 5.0
    knockdown_taken: float = -5.0
    punch_landed: float = 0.1
    star_bonus: float = 0.5
    heart_loss: float = 0.0
    score_weight: float = 0.01


@dataclass
class PPOConfig:
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class TrainConfig:
    total_timesteps: int = 10_000_000
    eval_freq: int = 50_000
    n_eval_episodes: int = 5
    save_freq: int = 100_000
    log_dir: str = "logs/"
    model_dir: str = "models/"
    video_dir: str = "videos/"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
