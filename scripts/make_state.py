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
    import gymnasium as gym
    from stable_baselines3 import PPO
    from config import Config
    from envs.wrappers import KnockdownRecovery, StochasticFrameSkip, PunchOutDiscretizer

    print(f"Loading model from {model_path}...")
    config = Config()

    # Build a stripped-down env for state generation:
    #   - KnockdownRecovery: pulses START+A to advance post-fight screens
    #   - StochasticFrameSkip: with sticky_prob=0 for reliable model play
    #   - PunchOutDiscretizer: so model.predict() actions work correctly
    #   - NO PunchOutRewardWrapper: removes Mac KO termination, which fires
    #     on false positives during the Glass Joe → Von Kaiser RAM transition
    #     (health_mac briefly reads 0 while health_com is non-zero)
    # Episodes only end via the large TimeLimit, so the loop controls flow.
    env = retro.make(game=game, state="Match1", render_mode=None)
    env = KnockdownRecovery(env)
    env = StochasticFrameSkip(env, n_frames=config.env.frame_skip, sticky_prob=0.0)
    env = PunchOutDiscretizer(env)
    if config.env.grayscale:
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=config.env.resize)
    env = gym.wrappers.FrameStackObservation(env, stack_size=config.env.frame_stack)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100_000)

    model = PPO.load(model_path)
    model = PPO.load(model_path)

    print("Playing through Glass Joe to find Match2 start...")
    # Knockdowns temporarily set health_com=0 in RAM; require it to stay
    # at 0 for this many consecutive steps to confirm a real KO.
    # Knockdown counts last up to 10 seconds = ~150 steps at 4-frame skip.
    # Require health_com==0 for longer than any possible knockdown count
    # to confirm a real KO (health stays at 0 through the whole transition).
    KO_CONFIRM_STEPS = 200  # ~13 seconds, safely above any knockdown count
    # After KO confirmed, require new opponent health to be stable at a
    # non-zero value for this many steps before saving.
    STABILITY_REQUIRED = 20

    max_attempts = 20
    for attempt in range(1, max_attempts + 1):
        obs, info = env.reset()
        glass_joe_beaten = False
        zero_count = 0
        stable_count = 0
        last_health_com_raw = -1

        print(f"  Attempt {attempt}...")
        for step in range(timeout):
            # Once KO is confirmed, press NOOP (action index 0) so Mac doesn't
            # fight Von Kaiser and risk getting KO'd before we can save the state.
            # Must pass integer 0 (not raw NOOP array) since PunchOutDiscretizer
            # is the top wrapper and expects discrete action indices.
            if glass_joe_beaten:
                action = 0
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            ram            = env.unwrapped.get_ram()
            health_com_ram = int(ram[ADDR_HEALTH_COM])
            health_mac_ram = int(ram[ADDR_HEALTH_MAC])
            clock_ram      = int(ram[ADDR_CLOCK_ACTIVE])

            # Without PunchOutRewardWrapper, the game loops forever if Mac is KO'd
            # (KnockdownRecovery keeps pressing START to continue the fight).
            # Detect Mac KO via RAM and break the attempt early.
            if not glass_joe_beaten and health_mac_ram == 0 and clock_ram == 1:
                print(f"  Attempt {attempt}: Mac KO'd at step {step}, retrying...")
                break

            # Confirm KO: health_com must stay at 0 for KO_CONFIRM_STEPS
            # consecutive steps (knockdowns recover; KOs don't)
            if not glass_joe_beaten:
                if health_com_ram == 0:
                    zero_count += 1
                    if zero_count >= KO_CONFIRM_STEPS:
                        glass_joe_beaten = True
                        stable_count = 0
                        last_health_com_raw = -1
                        print(f"  Attempt {attempt}: Glass Joe KO confirmed at step {step}")
                else:
                    zero_count = 0

            # Wait for new opponent: clock active, both fighters at full health.
            # Requiring opponent health == FULL_HEALTH (96) prevents false triggers
            # from Glass Joe recovering after a knockdown (reduced health, stable).
            if glass_joe_beaten:
                if clock_ram == 1 and health_mac_ram == FULL_HEALTH and health_com_ram == FULL_HEALTH:
                    if health_com_ram == last_health_com_raw:
                        stable_count += 1
                    else:
                        stable_count = 1
                        last_health_com_raw = health_com_ram

                    if stable_count >= STABILITY_REQUIRED:
                        state = env.unwrapped.em.get_state()
                        env.close()
                        print(f"  Next fight confirmed at step {step} "
                              f"(opponent health={health_com_ram} [full={FULL_HEALTH}], "
                              f"stable for {stable_count} steps)")
                        return state
                else:
                    stable_count = 0

            if terminated or truncated:
                if not glass_joe_beaten:
                    print(f"  Attempt {attempt}: episode ended without KO confirmed, retrying...")
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
