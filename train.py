"""Train a PPO agent to play Punch-Out!! NES."""

import argparse
import os
from datetime import datetime

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from config import Config
from envs.wrappers import make_env


MILESTONES = {
    0.01: "early",
    0.25: "quarter",
    0.50: "mid",
    0.75: "three-quarter",
    1.00: "final",
}


class VideoRecordingCallback(BaseCallback):
    """Records one episode at each training milestone and saves as MP4."""

    def __init__(self, config: Config, total_timesteps: int):
        super().__init__()
        self.config = config
        self.total_timesteps = total_timesteps
        self._remaining = dict(MILESTONES)  # fraction -> label
        self._pending: list[str] = []       # labels queued for next rollout end

    def _on_step(self) -> bool:
        # Only set flags here — recording mid-rollout causes a CUDA deadlock
        progress = self.num_timesteps / self.total_timesteps
        for fraction in list(self._remaining):
            if progress >= fraction:
                self._pending.append(self._remaining.pop(fraction))
        return True

    def _on_rollout_end(self) -> None:
        # GPU is idle between rollout and update — safe to run inference
        for label in self._pending:
            self._record(label)
        self._pending.clear()

    def _record(self, label: str) -> None:
        env = make_env(self.config, render_mode="rgb_array")()
        obs, _ = env.reset()
        frames, done = [], False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated
        env.close()

        if frames:
            os.makedirs(self.config.train.video_dir, exist_ok=True)
            path = os.path.join(
                self.config.train.video_dir,
                f"{label}_{self.num_timesteps}.mp4",
            )
            imageio.mimwrite(path, frames, fps=60)
            print(f"  [video] {label} ({self.num_timesteps:,} steps) → {path}")


def linear_schedule(initial_value: float):
    """Linear learning rate schedule from initial_value to 0."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Punch-Out!! NES")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (overrides config)")
    parser.add_argument("--state", type=str, default=None,
                        help="Game state to start from (e.g. Match1)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model checkpoint to resume from")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=None,
                        help="Initial learning rate (overrides config); use a lower value when fine-tuning")
    parser.add_argument("--clip-range", type=float, default=None,
                        help="PPO clip range (overrides config); smaller values (e.g. 0.05) for fine-tuning")
    parser.add_argument("--dummy-vec", action="store_true",
                        help="Use DummyVecEnv instead of SubprocVecEnv")
    args = parser.parse_args()

    config = Config()
    if args.state:
        config.env.state = args.state
    if args.n_envs:
        config.env.n_envs = args.n_envs
    if args.lr:
        config.ppo.learning_rate = args.lr
    if args.clip_range:
        config.ppo.clip_range = args.clip_range

    total_timesteps = args.timesteps or config.train.total_timesteps

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_model_dir = os.path.join(config.train.model_dir, run_id)
    run_video_dir = os.path.join(config.train.video_dir, run_id)

    os.makedirs(config.train.log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)

    # Training envs
    vec_cls = DummyVecEnv if args.dummy_vec else SubprocVecEnv
    train_envs = VecMonitor(
        vec_cls([make_env(config) for _ in range(config.env.n_envs)])
    )

    # Eval env (single subprocess so main process stays emulator-free for video recording)
    eval_env = VecMonitor(SubprocVecEnv([make_env(config, eval_env=True)]))

    # Per-run video config so the callback writes to the right directory
    run_config = Config()
    run_config.__dict__.update(config.__dict__)
    run_config.train.video_dir = run_video_dir

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_model_dir, "best"),
        log_path=config.train.log_dir,
        eval_freq=max(config.train.eval_freq // config.env.n_envs, 1),
        n_eval_episodes=config.train.n_eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.train.save_freq // config.env.n_envs, 1),
        save_path=os.path.join(run_model_dir, "checkpoints"),
        name_prefix="punchout_ppo",
    )
    video_callback = VideoRecordingCallback(run_config, total_timesteps)

    # Model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_envs, tensorboard_log=config.train.log_dir)
        model.learning_rate = linear_schedule(config.ppo.learning_rate)
        model.clip_range = lambda _: config.ppo.clip_range
    else:
        model = PPO(
            "CnnPolicy",
            train_envs,
            learning_rate=linear_schedule(config.ppo.learning_rate),
            n_steps=config.ppo.n_steps,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_range=config.ppo.clip_range,
            ent_coef=config.ppo.ent_coef,
            vf_coef=config.ppo.vf_coef,
            max_grad_norm=config.ppo.max_grad_norm,
            tensorboard_log=config.train.log_dir,
            verbose=1,
        )

    print(f"Run ID: {run_id}")
    print(f"Training for {total_timesteps:,} timesteps with {config.env.n_envs} envs")
    print(f"Game: {config.env.game}, State: {config.env.state}")
    print(f"Action space: {train_envs.action_space}")
    print(f"Observation space: {train_envs.observation_space}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, video_callback],
        tb_log_name="punchout_ppo",
    )

    final_path = os.path.join(run_model_dir, "punchout_ppo_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
