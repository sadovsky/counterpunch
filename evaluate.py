"""Evaluate a trained PPO agent on Punch-Out!! and optionally record video."""

import argparse
import os

import imageio
import numpy as np
from stable_baselines3 import PPO

from config import Config
from envs.wrappers import make_env

# RAM addresses for debug output
_ADDR_FIGHT_INIT   = 0
_ADDR_MATCH_ID     = 1
_ADDR_OPP_TYPE     = 2
_ADDR_ROUND        = 6
_ADDR_FIGHT_STATE  = 4
_ADDR_FIGHT_ID     = 8
_ADDR_CLOCK_ACTIVE = 768
_ADDR_HEALTH_MAC   = 913
_ADDR_HEALTH_COM   = 920


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
        states = [args.state]
    else:
        states = [config.env.state] + list(config.env.generalization_states)

    render_mode = "human" if args.render else "rgb_array"
    model = PPO.load(args.model)
    os.makedirs(config.train.video_dir, exist_ok=True)

    for state in states:
        config.env.state = state
        env_fn = make_env(config, render_mode=render_mode, eval_env=True)
        env = env_fn()
        print(f"\n=== State: {state} ===")

        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            frames = []

            prev_ram_state = {}

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                done = terminated or truncated

                # Read RAM for debug output
                try:
                    ram = env.unwrapped.get_ram()
                    ram_state = {
                        "fight_init":  int(ram[_ADDR_FIGHT_INIT]),
                        "match_id":    int(ram[_ADDR_MATCH_ID]),
                        "opp_type":    int(ram[_ADDR_OPP_TYPE]),
                        "round":       int(ram[_ADDR_ROUND]),
                        "fight_state": int(ram[_ADDR_FIGHT_STATE]),
                        "fight_id":    int(ram[_ADDR_FIGHT_ID]),
                        "clock":       int(ram[_ADDR_CLOCK_ACTIVE]),
                        "health_mac":  int(ram[_ADDR_HEALTH_MAC]),
                        "health_com":  int(ram[_ADDR_HEALTH_COM]),
                    }
                    if ram_state != prev_ram_state:
                        print(
                            f"  step={step_count:5d}  "
                            f"match={ram_state['match_id']}  opp_type={ram_state['opp_type']}  "
                            f"round={ram_state['round']}  id={ram_state['fight_id']:3d}  "
                            f"fight=0x{ram_state['fight_state']:02X}  clk={ram_state['clock']}  "
                            f"mac={ram_state['health_mac']:3d}  com={ram_state['health_com']:3d}  "
                            f"r={reward:+.2f}",
                            flush=True,
                        )
                        prev_ram_state = ram_state
                except Exception:
                    pass

                if step_count % 50 == 0 and ram_state == prev_ram_state:
                    print(f"  step={step_count:5d}  (no change)", flush=True)

                if args.record:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

            print(f"Episode {ep + 1}: reward={total_reward:.1f}, steps={step_count}")

            if args.record and frames:
                video_path = os.path.join(config.train.video_dir, f"{state}_episode_{ep + 1}.mp4")
                imageio.mimwrite(video_path, frames, fps=30)
                print(f"  Video saved to {video_path}")

        env.close()


if __name__ == "__main__":
    main()
