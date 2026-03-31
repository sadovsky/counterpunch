"""Evaluate a trained PPO agent on Punch-Out!! and optionally record video."""

import argparse
import os

import imageio
import numpy as np
from stable_baselines3 import PPO

from config import Config
from envs.wrappers import make_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate Punch-Out!! agent")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.zip)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--state", type=str, default=None,
                        help="Game state to evaluate on")
    parser.add_argument("--record", action="store_true",
                        help="Record video of episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render live (human mode)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions")
    args = parser.parse_args()

    config = Config()
    if args.state:
        config.env.state = args.state

    render_mode = "human" if args.render else "rgb_array"
    env_fn = make_env(config, render_mode=render_mode)
    env = env_fn()

    model = PPO.load(args.model)

    os.makedirs(config.train.video_dir, exist_ok=True)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            if args.record:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        print(f"Episode {ep + 1}: reward={total_reward:.1f}, steps={step_count}")

        if args.record and frames:
            video_path = os.path.join(config.train.video_dir, f"episode_{ep + 1}.mp4")
            imageio.mimwrite(video_path, frames, fps=30)
            print(f"  Video saved to {video_path}")

    env.close()


if __name__ == "__main__":
    main()
