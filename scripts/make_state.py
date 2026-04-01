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


# RAM addresses (see MEMORY_MAP.md and DataCrystal RAM map)
ADDR_OPPONENT     = 1     # 0x0001 — current opponent ID (0=Glass Joe, 1=Von Kaiser, …)
ADDR_FIGHT_STATE  = 4     # 0x0004 — 0xFF=fight active, 0x01=between rounds
ADDR_CLOCK_ACTIVE = 768   # 0x0300 — 1 when a fight is live
ADDR_HEALTH_MAC   = 913   # 0x0391 — player health (max 96)
ADDR_HEALTH_COM   = 920   # 0x0398 — opponent health
FULL_HEALTH       = 96    # 0x60

FIGHT_ACTIVE      = 0xFF
VON_KAISER_ID     = 1

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

    print("Playing through Glass Joe to find Match2 start...")
    # Detection approach: watch for the new-fight reset signal.
    #
    # When the model KOs Glass Joe, health_com drops to 0 (knockdown) and stays
    # there through the TKO animation. Once the next fight loads, BOTH fighters
    # reset to FULL_HEALTH simultaneously (mac=96, com=96).
    #
    # We use a simple state machine:
    #   1. Wait for health_com == 0 (Glass Joe knocked down/KO'd)
    #   2. After that, wait for BOTH fighters at FULL_HEALTH for STABILITY_REQUIRED
    #      consecutive steps — that's the new fight fully loaded
    #
    # This avoids relying on opponent_id (which doesn't update reliably) and
    # avoids the old 200-step consecutive-zero threshold.
    STABILITY_REQUIRED = 30

    max_attempts = 20
    for attempt in range(1, max_attempts + 1):
        obs, info = env.reset()
        knockdown_seen = False   # health_com has hit 0 at least once
        new_fight_seen = False   # both fighters reset to full health after knockdown
        stable_count   = 0
        prev_fight_state  = -1
        prev_health_mac   = -1
        prev_health_com   = -1

        print(f"  Attempt {attempt}/{max_attempts}")
        for step in range(timeout):
            ram            = env.unwrapped.get_ram()
            opponent_id    = int(ram[ADDR_OPPONENT])
            fight_state    = int(ram[ADDR_FIGHT_STATE])
            health_mac_ram = int(ram[ADDR_HEALTH_MAC])
            health_com_ram = int(ram[ADDR_HEALTH_COM])

            # Print on any state change, plus a heartbeat every 50 steps
            state_changed = (
                fight_state    != prev_fight_state or
                health_mac_ram != prev_health_mac  or
                health_com_ram != prev_health_com
            )
            if state_changed or step % 50 == 0:
                tag = ""
                if stable_count > 0:
                    tag = f" [stable {stable_count}/{STABILITY_REQUIRED}]"
                elif new_fight_seen:
                    tag = " [waiting for fight_state=0xFF]"
                elif knockdown_seen:
                    tag = " [waiting for full-health reset]"
                print(f"    step={step:4d}  opp={opponent_id}  "
                      f"fight=0x{fight_state:02X}  mac={health_mac_ram:3d}  "
                      f"com={health_com_ram:3d}{tag}", flush=True)
            prev_fight_state = fight_state
            prev_health_mac  = health_mac_ram
            prev_health_com  = health_com_ram

            # fight_state != 0xFF means Mac is down (opponent knockdowns
            # leave fight_state=0xFF, so this unambiguously means Mac is down).
            # Retry rather than trying to mash Mac back up.
            if not knockdown_seen and fight_state != FIGHT_ACTIVE:
                print(f"  Attempt {attempt}: Mac knocked down at step {step}, retrying...")
                break

            # Stage 1: watch for Glass Joe going down (health_com → 0)
            if health_com_ram == 0:
                knockdown_seen = True

            # Stage 2: after knockdown, watch for both fighters resetting to
            # FULL_HEALTH simultaneously — that's the new fight loading
            if knockdown_seen and not new_fight_seen:
                if health_mac_ram == FULL_HEALTH and health_com_ram == FULL_HEALTH:
                    print(f"  Attempt {attempt}: new fight detected at step {step} "
                          f"(opp={opponent_id}, fight=0x{fight_state:02X})")
                    new_fight_seen = True

            # Stage 3: wait for fight to be fully active, then save
            if new_fight_seen:
                if fight_state == FIGHT_ACTIVE and health_mac_ram == FULL_HEALTH and health_com_ram > 0:
                    stable_count += 1
                    if stable_count >= STABILITY_REQUIRED:
                        state = env.unwrapped.em.get_state()
                        env.close()
                        print(f"  Match2 state saved at step {step} "
                              f"(opponent health={health_com_ram}, "
                              f"stable {stable_count} steps)")
                        return state
                else:
                    stable_count = 0

            # Press NOOP once new fight is detected so Mac doesn't fight Von Kaiser
            if new_fight_seen:
                action = 0
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                stage = "new fight found but" if new_fight_seen else (
                    "Glass Joe down, new fight not found," if knockdown_seen
                    else "Glass Joe not beaten,")
                print(f"  Attempt {attempt}: {stage} episode ended at step {step}, retrying...")
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
