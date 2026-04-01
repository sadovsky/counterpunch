"""Generate fight save states for the currently installed ROM.

Match1: boots from power-on and saves the moment the Glass Joe fight begins.
Match2: loads Match1, runs the trained model through Glass Joe, and saves the
        moment the next fight (Von Kaiser) begins.

Usage:
    python scripts/make_state.py
    python scripts/make_state.py --match 2 --model models/.../best/best_model.zip
    python scripts/make_state.py --game PunchOut-Nes-v0 --state Match1 --timeout 30000
"""

import argparse
import gzip
import os
import sys

import numpy as np
import stable_retro as retro


# RAM addresses (see MEMORY_MAP.md)
ADDR_CLOCK_ACTIVE = 768   # 0x0300 — 1 when a fight is live
ADDR_HEALTH_MAC   = 913   # 0x0391 — player health (max 96)
ADDR_HEALTH_COM   = 920   # 0x0398 — opponent health (max 96)
FULL_HEALTH       = 96    # 0x60

START = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
NOOP  = np.zeros(9, dtype=np.int8)


def find_match1(game: str, timeout: int) -> bytes:
    """Boot from power-on and capture the state at the start of the Glass Joe fight."""
    env = retro.make(game, state=retro.State.NONE, render_mode=None)
    env.reset()

    print(f"Booting {game}, navigating menus (up to {timeout} frames)...")
    for frame in range(timeout):
        action = START if frame % 60 < 5 else NOOP
        env.step(action)

        ram = env.get_ram()
        clock      = int(ram[ADDR_CLOCK_ACTIVE])
        health_mac = int(ram[ADDR_HEALTH_MAC])
        health_com = int(ram[ADDR_HEALTH_COM])

        if clock == 1 and health_mac == FULL_HEALTH and health_com == FULL_HEALTH:
            state = env.em.get_state()
            env.close()
            print(f"  Fight detected at frame {frame} ({frame / 60:.1f}s)")
            return state

    env.close()
    raise RuntimeError(
        f"Fight not detected within {timeout} frames. "
        "Try increasing --timeout or check your ROM."
    )


def find_match2(game: str, model_path: str, timeout: int) -> bytes:
    """Load Match1, beat Glass Joe with the trained model, save at the next fight."""
    # Import here so make_state.py has no mandatory dependency on SB3 / project code
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from stable_baselines3 import PPO
    from config import Config
    from envs.wrappers import make_env

    print(f"Loading model from {model_path}...")
    config = Config()
    config.env.state = "Match1"
    env = make_env(config)()
    model = PPO.load(model_path)

    print("Playing through Glass Joe to find Match2 start...")
    # Require opponent health to hold the same non-zero value for this many
    # consecutive steps after the KO — transition screens flicker, a real
    # fight start is stable.
    STABILITY_REQUIRED = 15

    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        obs, info = env.reset()
        glass_joe_beaten = False
        stable_count = 0
        last_health_com_raw = -1

        for step in range(timeout):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            health_com = info.get("health_com", FULL_HEALTH)

            if not glass_joe_beaten and health_com == 0:
                glass_joe_beaten = True
                stable_count = 0
                last_health_com_raw = -1
                print(f"  Attempt {attempt}: Glass Joe KO at step {step}")

            if glass_joe_beaten:
                ram = env.unwrapped.get_ram()
                clock          = int(ram[ADDR_CLOCK_ACTIVE])
                health_mac_raw = int(ram[ADDR_HEALTH_MAC])
                health_com_raw = int(ram[ADDR_HEALTH_COM])

                if clock == 1 and health_mac_raw == FULL_HEALTH and health_com_raw > 0:
                    if health_com_raw == last_health_com_raw:
                        stable_count += 1
                    else:
                        stable_count = 1
                        last_health_com_raw = health_com_raw

                    if stable_count >= STABILITY_REQUIRED:
                        state = env.unwrapped.em.get_state()
                        env.close()
                        print(f"  Next fight confirmed at step {step} "
                              f"(opponent health={health_com_raw}, "
                              f"stable for {stable_count} steps)")
                        return state
                else:
                    stable_count = 0

            if terminated or truncated:
                if not glass_joe_beaten:
                    print(f"  Attempt {attempt}: episode ended without KO, retrying...")
                else:
                    print(f"  Attempt {attempt}: episode ended before next fight confirmed, retrying...")
                break

    env.close()
    raise RuntimeError(
        "Could not reach Match2 after {} attempts. "
        "Try a better model checkpoint or increase --timeout.".format(max_attempts)
    )


def get_data_dir(game: str) -> str:
    return os.path.join(
        os.path.dirname(retro.__file__),
        "data", "stable", game,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate fight save states for stable-retro")
    parser.add_argument("--match",   type=int, default=1,              help="Which match to generate (1 or 2)")
    parser.add_argument("--game",    default="PunchOut-Nes-v0",        help="stable-retro game ID")
    parser.add_argument("--state",   default=None,                     help="State name to write (default: MatchN)")
    parser.add_argument("--model",   default=None,                     help="Path to trained model (.zip), required for --match 2")
    parser.add_argument("--timeout", default=30000, type=int,          help="Max frames/steps to search")
    parser.add_argument("--output",  default=None,                     help="Override output path")
    args = parser.parse_args()

    state_name = args.state or f"Match{args.match}"

    if args.match == 1:
        fight_state = find_match1(args.game, args.timeout)
    elif args.match == 2:
        if not args.model:
            parser.error("--match 2 requires --model <path/to/model.zip>")
        fight_state = find_match2(args.game, args.model, args.timeout)
    else:
        parser.error("--match must be 1 or 2")

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(get_data_dir(args.game), f"{state_name}.state")

    with gzip.open(out_path, "wb") as f:
        f.write(fight_state)
    print(f"  Saved → {out_path}")

    # Sanity check (Match1 only — Match2 opponent health varies)
    if args.match == 1:
        print("Verifying...")
        env = retro.make(args.game, state=state_name, render_mode=None)
        env.reset()
        for _ in range(10):
            _, _, _, _, info = env.step(NOOP)
        env.close()
        mac = info.get("health_mac", "?")
        com = info.get("health_com", "?")
        print(f"  health_mac={mac}  health_com={com}")
        if mac == FULL_HEALTH and com == FULL_HEALTH:
            print("  OK — state loads correctly at full health.")
        else:
            print("  WARNING — health values unexpected. State may be misaligned.")


if __name__ == "__main__":
    main()
