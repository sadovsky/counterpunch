"""Generate a Match1.state save file for the currently installed ROM.

Boots the game from power-on, navigates through menus by pressing START,
and saves a gzip-compressed emulator snapshot the moment a fight begins
(clock active, both fighters at full health).

Usage:
    python scripts/make_state.py
    python scripts/make_state.py --game PunchOut-Nes-v0 --state Match1 --timeout 30000
"""

import argparse
import gzip
import os

import numpy as np
import stable_retro as retro


# RAM addresses (see MEMORY_MAP.md)
ADDR_CLOCK_ACTIVE = 768   # 0x0300 — 1 when a fight is live
ADDR_HEALTH_MAC   = 913   # 0x0391 — player health (max 96)
ADDR_HEALTH_COM   = 920   # 0x0398 — opponent health (max 96)
FULL_HEALTH       = 96    # 0x60

START = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
NOOP  = np.zeros(9, dtype=np.int8)


def find_fight(game: str, timeout: int) -> bytes:
    env = retro.make(game, state=retro.State.NONE, render_mode=None)
    env.reset()

    print(f"Booting {game}, navigating menus (up to {timeout} frames)...")
    for frame in range(timeout):
        action = START if frame % 60 < 5 else NOOP
        env.step(action)

        ram = env.get_ram()
        clock  = int(ram[ADDR_CLOCK_ACTIVE])
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


def get_data_dir(game: str) -> str:
    data_dir = os.path.join(
        os.path.dirname(retro.__file__),
        "data", "stable", game,
    )
    return data_dir


def main():
    parser = argparse.ArgumentParser(description="Generate a fight save state for stable-retro")
    parser.add_argument("--game",    default="PunchOut-Nes-v0", help="stable-retro game ID")
    parser.add_argument("--state",   default="Match1",          help="State name to write")
    parser.add_argument("--timeout", default=30000, type=int,   help="Max frames to search")
    parser.add_argument("--output",  default=None,              help="Override output path")
    args = parser.parse_args()

    fight_state = find_fight(args.game, args.timeout)

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(get_data_dir(args.game), f"{args.state}.state")

    with gzip.open(out_path, "wb") as f:
        f.write(fight_state)

    print(f"  Saved → {out_path}")

    # Quick sanity check
    print("Verifying...")
    env = retro.make(args.game, state=args.state, render_mode=None)
    _, info = env.reset()
    env.step(NOOP)
    _, info = env.reset()
    for _ in range(10):
        _, _, _, _, info = env.step(NOOP)
    env.close()

    mac = info.get("health_mac", "?")
    com = info.get("health_com", "?")
    clock = info.get("clock_active", "?")
    print(f"  health_mac={mac}  health_com={com}")
    if mac == FULL_HEALTH and com == FULL_HEALTH:
        print("  OK — state loads correctly at full health.")
    else:
        print("  WARNING — health values unexpected. State may be misaligned.")


if __name__ == "__main__":
    main()
