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


# RAM addresses (see MEMORY_MAP.md, DataCrystal, and TASVideos RAM maps)
ADDR_FIGHT_INIT   = 0     # 0x0000 — 1=fight started, 0=not (cut scene / menu)
ADDR_FIGHT_STATE  = 4     # 0x0004 — 0xFF=in fight, 0x01=between rounds or cut scene
ADDR_FIGHT_ID     = 8     # 0x0008 — identifies current opponent (Von Kaiser=32)
ADDR_CLOCK_ACTIVE = 768   # 0x0300 — 1 when fight clock is running
ADDR_HEALTH_MAC   = 913   # 0x0391 — player health (max 96)
ADDR_HEALTH_COM   = 920   # 0x0398 — opponent health
FULL_HEALTH       = 96    # 0x60

FIGHT_ACTIVE      = 0xFF
VON_KAISER_FIGHT_ID = 32  # 0x20 — fight ID for Von Kaiser (from TASVideos RAM map)

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
    from envs.wrappers import StochasticFrameSkip, PunchOutDiscretizer

    print(f"Loading model from {model_path}...")
    config = Config()

    # Build a stripped-down env for state generation:
    #   - NO KnockdownRecovery: its START+A pulsing fires whenever clk=0,
    #     which includes the between-rounds break. START pauses the NES game,
    #     causing the break screen to freeze indefinitely. We handle all
    #     transitions manually in the loop below.
    #   - StochasticFrameSkip: with sticky_prob=0 for reliable model play
    #   - PunchOutDiscretizer: so model.predict() actions work correctly
    #   - NO PunchOutRewardWrapper: removes Mac KO termination which fires
    #     on false positives during the Glass Joe → Von Kaiser RAM transition
    # Episodes only end via the large TimeLimit, so the loop controls flow.
    env = retro.make(game=game, state="Match1", render_mode=None)
    env = StochasticFrameSkip(env, n_frames=config.env.frame_skip, sticky_prob=0.0)
    env = PunchOutDiscretizer(env)
    if config.env.grayscale:
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=config.env.resize)
    env = gym.wrappers.FrameStackObservation(env, stack_size=config.env.frame_stack)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100_000)

    model = PPO.load(model_path)

    print("Playing through Glass Joe to find Match2 start...")
    # Use fight_id (0x0008) to detect Von Kaiser's fight. Von Kaiser = 32.
    # This is more reliable than health polling: fight_id is set exactly when
    # the new fight loads, independent of health values or round transitions.
    #
    # Between rounds and during knockdowns, fight_state = 0x01. The NES
    # knockdown-recovery screen needs directional button mashing (LEFT/RIGHT),
    # not START — so we mash those when fight_state != 0xFF instead of breaking.
    # This lets the fight run through multiple rounds if needed.
    STABILITY_REQUIRED = 30

    max_attempts = 50
    for attempt in range(1, max_attempts + 1):
        obs, info = env.reset()
        von_kaiser_seen = False
        stable_count    = 0
        prev_fight_state = -1
        prev_fight_id    = -1
        prev_clock       = -1
        prev_health_mac  = -1
        prev_health_com  = -1

        print(f"  Attempt {attempt}/{max_attempts}")
        for step in range(timeout):
            ram            = env.unwrapped.get_ram()
            fight_init     = int(ram[ADDR_FIGHT_INIT])
            fight_state    = int(ram[ADDR_FIGHT_STATE])
            fight_id       = int(ram[ADDR_FIGHT_ID])
            clock_ram      = int(ram[ADDR_CLOCK_ACTIVE])
            health_mac_ram = int(ram[ADDR_HEALTH_MAC])
            health_com_ram = int(ram[ADDR_HEALTH_COM])

            # Print on any state change, plus a heartbeat every 50 steps
            state_changed = (
                fight_state    != prev_fight_state or
                fight_id       != prev_fight_id    or
                clock_ram      != prev_clock       or
                health_mac_ram != prev_health_mac  or
                health_com_ram != prev_health_com
            )
            if state_changed or step % 50 == 0:
                tag = ""
                if stable_count > 0:
                    tag = f" [stable {stable_count}/{STABILITY_REQUIRED}]"
                elif von_kaiser_seen:
                    tag = " [waiting for fight active]"
                print(f"    step={step:4d}  init={fight_init}  id={fight_id:3d}  "
                      f"fight=0x{fight_state:02X}  clk={clock_ram}  "
                      f"mac={health_mac_ram:3d}  com={health_com_ram:3d}{tag}", flush=True)
            prev_fight_state = fight_state
            prev_fight_id    = fight_id
            prev_clock       = clock_ram
            prev_health_mac  = health_mac_ram
            prev_health_com  = health_com_ram

            # Cheat: keep Glass Joe's health at 1 (any punch finishes him)
            # and clamp Mac's health so he can't be KO'd before we save.
            # Must write both "next" and "current" health addresses — the game
            # uses both (next=913/920, current=914/921 per DataCrystal RAM map).
            # Only active while still fighting Glass Joe (not Von Kaiser).
            # Mac's health resets to full at the start of each new fight, so
            # the saved Von Kaiser state will still have mac=96.
            if not von_kaiser_seen and fight_state == FIGHT_ACTIVE:
                ram[ADDR_HEALTH_COM]     = 1   # 920 — opponent next health
                ram[ADDR_HEALTH_COM + 1] = 1   # 921 — opponent current health
                if health_mac_ram < 32:
                    ram[ADDR_HEALTH_MAC]     = FULL_HEALTH  # 913 — mac next health
                    ram[ADDR_HEALTH_MAC + 1] = FULL_HEALTH  # 914 — mac current health

            # Detect Von Kaiser fight by fight_id
            if fight_id == VON_KAISER_FIGHT_ID:
                if not von_kaiser_seen:
                    print(f"  Attempt {attempt}: Von Kaiser (fight_id=32) detected "
                          f"at step {step}")
                    von_kaiser_seen = True

                # Wait for fight fully active with Mac at full health, then save
                if fight_state == FIGHT_ACTIVE and health_mac_ram == FULL_HEALTH and health_com_ram > 0:
                    stable_count += 1
                    if stable_count >= STABILITY_REQUIRED:
                        state = env.unwrapped.em.get_state()
                        env.close()
                        print(f"  Match2 state saved at step {step} "
                              f"(Von Kaiser health={health_com_ram}, "
                              f"stable {stable_count} steps)")
                        return state
                else:
                    stable_count = 0

            # Action selection:
            #   - Von Kaiser detected: NOOP so Mac doesn't fight him
            #   - fight_state != 0xFF (between rounds / Mac down): mash LEFT+RIGHT
            #     to fill the get-up meter, plus occasional START for cut scenes
            #   - Otherwise: let the model play
            if von_kaiser_seen:
                action = 0  # NOOP — don't fight Von Kaiser
            elif fight_state != FIGHT_ACTIVE:
                if fight_init == 1:
                    # Mac knocked down mid-round — mash A and B to fill get-up meter
                    action = 4 if step % 2 == 0 else 5  # A=4, B=5
                else:
                    # Between rounds / victory screens / cut scenes.
                    # The between-rounds break advances automatically (don't press
                    # START — it pauses the NES). Pulse START slowly (once per
                    # 60 steps ≈ every 4 game-seconds) to advance any screens
                    # that do need manual input without triggering the pause menu.
                    action = 8 if step % 60 == 0 else 0
            else:
                action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                stage = "Von Kaiser found but" if von_kaiser_seen else "Glass Joe not beaten,"
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
