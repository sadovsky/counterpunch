"""Train a PPO agent to play Punch-Out!! NES."""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from config import Config
from envs.wrappers import make_env


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
    parser.add_argument("--dummy-vec", action="store_true",
                        help="Use DummyVecEnv instead of SubprocVecEnv")
    args = parser.parse_args()

    config = Config()
    if args.state:
        config.env.state = args.state
    if args.n_envs:
        config.env.n_envs = args.n_envs

    total_timesteps = args.timesteps or config.train.total_timesteps

    os.makedirs(config.train.log_dir, exist_ok=True)
    os.makedirs(config.train.model_dir, exist_ok=True)

    # Training envs
    vec_cls = DummyVecEnv if args.dummy_vec else SubprocVecEnv
    train_envs = VecMonitor(
        vec_cls([make_env(config) for _ in range(config.env.n_envs)])
    )

    # Eval env (single, for deterministic evaluation)
    eval_env = VecMonitor(DummyVecEnv([make_env(config)]))

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.train.model_dir, "best"),
        log_path=config.train.log_dir,
        eval_freq=max(config.train.eval_freq // config.env.n_envs, 1),
        n_eval_episodes=config.train.n_eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.train.save_freq // config.env.n_envs, 1),
        save_path=os.path.join(config.train.model_dir, "checkpoints"),
        name_prefix="punchout_ppo",
    )

    # Model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_envs, tensorboard_log=config.train.log_dir)
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

    print(f"Training for {total_timesteps:,} timesteps with {config.env.n_envs} envs")
    print(f"Game: {config.env.game}, State: {config.env.state}")
    print(f"Action space: {train_envs.action_space}")
    print(f"Observation space: {train_envs.observation_space}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="punchout_ppo",
    )

    final_path = os.path.join(config.train.model_dir, "punchout_ppo_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
