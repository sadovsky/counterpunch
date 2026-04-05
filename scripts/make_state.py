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
ADDR_MATCH_ID     = 1     # 0x0001 — match/opponent index: 0=Glass Joe, 1=Von Kaiser, ...
ADDR_OPP_TYPE     = 2     # 0x0002 — opponent ROM bank / type (0=Glass Joe per TASVideos)
ADDR_ROUND        = 6     # 0x0006 — current round number
ADDR_FIGHT_STATE  = 4     # 0x0004 — 0xFF=in fight, 0x01=between rounds or cut scene
ADDR_FIGHT_ID     = 8     # 0x0008 — per-round sub-ID (observed: 0/50 for Glass Joe rounds)
ADDR_CLOCK_ACTIVE = 768   # 0x0300 — 1 when fight clock is running
ADDR_HEALTH_MAC   = 913   # 0x0391 — player health (max 96)
ADDR_HEALTH_COM   = 920   # 0x0398 — opponent health
FULL_HEALTH       = 96    # 0x60

FIGHT_ACTIVE        = 0xFF
GLASS_JOE_MATCH_ID  = 0   # 0x0001 value for Glass Joe
VON_KAISER_MATCH_ID = 1   # 0x0001 value for Von Kaiser (DataCrystal: 00-0D standard circuits)

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


def find_match2(game: str, model_path: str, timeout: int, render: bool = False) -> bytes:
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
    #   - NO PunchOutRewardWrapper: unnecessary for state generation
    # Episodes only end via the large TimeLimit, so the loop controls flow.
    env = retro.make(game=game, state="Match1", render_mode="human" if render else None)
    env = StochasticFrameSkip(env, n_frames=config.env.frame_skip, sticky_prob=0.0)
    env = PunchOutDiscretizer(env)
    if config.env.grayscale:
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=config.env.resize)
    env = gym.wrappers.FrameStackObservation(env, stack_size=config.env.frame_stack)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100_000)

    model = PPO.load(model_path)

    print("Playing through Glass Joe to find Match2 start...")
    # Detect Glass Joe beaten: fight_state leaves 0xFF with com==0 (KO/TKO at
    # end of a round). Mid-round knockdowns keep fight_state=0xFF so they don't
    # trigger this. Once beaten, wait for the next stable full-health active
    # fight state — that's the Von Kaiser fight starting.
    STABILITY_REQUIRED = 30

    max_attempts = 50
    for attempt in range(1, max_attempts + 1):
        obs, info = env.reset()
        glass_joe_beaten = False
        stable_count     = 0
        prev_fight_state = -1
        prev_fight_id    = -1
        prev_clock       = -1
        prev_health_mac  = -1
        prev_health_com  = -1

        print(f"  Attempt {attempt}/{max_attempts}")
        for step in range(timeout):
            ram            = env.unwrapped.get_ram()
            fight_init     = int(ram[ADDR_FIGHT_INIT])
            match_id       = int(ram[ADDR_MATCH_ID])
            opp_type       = int(ram[ADDR_OPP_TYPE])
            round_num      = int(ram[ADDR_ROUND])
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
                elif glass_joe_beaten:
                    tag = " [waiting for next fight]"
                print(f"    step={step:4d}  match={match_id}  opp_type={opp_type}  "
                      f"round={round_num}  id={fight_id:3d}  "
                      f"fight=0x{fight_state:02X}  clk={clock_ram}  "
                      f"mac={health_mac_ram:3d}  com={health_com_ram:3d}{tag}", flush=True)
            prev_fight_state = fight_state
            prev_fight_id    = fight_id
            prev_clock       = clock_ram
            prev_health_mac  = health_mac_ram
            prev_health_com  = health_com_ram

            # Primary: detect Von Kaiser by match_id (0x0001) transitioning to 1.
            # Fallback: detect Glass Joe beaten via fight_state leaving 0xFF with
            # com==0 (end-of-round KO/TKO). Mid-round knockdowns keep fight_state=0xFF
            # so they don't trigger this.
            if not glass_joe_beaten:
                if match_id == VON_KAISER_MATCH_ID:
                    glass_joe_beaten = True
                    print(f"  Attempt {attempt}: Von Kaiser detected via match_id=1 "
                          f"at step {step} (opp_type={opp_type})")
                elif fight_state != FIGHT_ACTIVE and health_com_ram == 0:
                    glass_joe_beaten = True
                    print(f"  Attempt {attempt}: Glass Joe beaten (fight_state KO) "
                          f"at step {step} (match_id={match_id}, fight_id={fight_id})")

            # Once Glass Joe is beaten, wait for the next fight to stabilize at
            # full health before saving.
            if glass_joe_beaten:
                if fight_state == FIGHT_ACTIVE and health_mac_ram == FULL_HEALTH and health_com_ram > 0:
                    stable_count += 1
                    if stable_count >= STABILITY_REQUIRED:
                        state = env.unwrapped.em.get_state()
                        env.close()
                        print(f"  Match2 state saved at step {step} "
                              f"(match_id={match_id}, opp_type={opp_type}, "
                              f"opponent health={health_com_ram}, "
                              f"stable {stable_count} steps)")
                        return state
                else:
                    stable_count = 0

            # Action selection:
            #   - Glass Joe beaten: NOOP so Mac doesn't fight the next opponent
            #   - fight_state != 0xFF (between rounds / Mac down): mash A+B or
            #     pulse START depending on context
            #   - Otherwise: let the model play
            if glass_joe_beaten:
                if fight_state != FIGHT_ACTIVE:
                    # Advance victory/intro screens with slow START pulse
                    action = 8 if step % 60 == 0 else 0
                else:
                    action = 0  # NOOP — Von Kaiser fight active, don't engage
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
                stage = "Glass Joe beaten but" if glass_joe_beaten else "Glass Joe not beaten,"
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
    parser.add_argument("--render",  action="store_true",              help="Render the game window while running")
    args = parser.parse_args()

    state_name = args.state or f"Match{args.match}"

    if args.match == 1:
        fight_state = find_match1(args.game, args.timeout)
    elif args.match == 2:
        if not args.model:
            parser.error("--match 2 requires --model <path/to/model.zip>")
        fight_state = find_match2(args.game, args.model, args.timeout, render=args.render)
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
