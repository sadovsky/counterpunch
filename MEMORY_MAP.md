# Finding RAM Addresses and Building a Fight State for NES Punch-Out!!

When training a reinforcement learning agent to play a game, one of the first problems you hit is: *how does the agent know what's happening?* Pixel observations can work on their own, but shaped rewards — penalizing the agent for taking damage, rewarding it for landing punches — require reading structured game state out of RAM. For NES Punch-Out!!, that meant figuring out exactly where the emulator stores health, score, and hearts in its 2048 bytes of working memory.

This is a walkthrough of how we solved that problem for a non-standard ROM dump using [stable-retro](https://github.com/Farama-Foundation/stable-retro).

---

## The Problem: Wrong ROM, Wrong Addresses

stable-retro ships with game data for Punch-Out!! (`PunchOut-Nes-v0`), including a `data.json` that maps RAM addresses to named variables:

```json
{
  "info": {
    "health_mac": { "address": 913, "type": "|u1" },
    "health_com": { "address": 920, "type": "|u1" },
    "heart":      { "address": 801, "type": ">n4" },
    "score":      { "address": 1000, "type": ">n6" }
  }
}
```

It also ships with a `Match1.state` — a gzip-compressed emulator snapshot that drops the game directly into a fight against Glass Joe.

The catch: both of these were built against a specific ROM dump (SHA1 `fa6222e73f910010b9cacf023d575e7e7b94e84a`). Our ROM had a different hash. When we loaded `Match1.state` with the wrong ROM, the emulator came up in a corrupted state. Every `info` value read zero. Every step was garbage.

```
Step 1: info={'health_com': 0, 'health_mac': 0, 'heart': 0, 'score': 0}
```

---

## Verifying the ROM Boots at All

The first thing to establish was whether the ROM itself was functional, ignoring the state file. Using `retro.State.NONE` tells stable-retro to boot from power-on instead of loading a save state:

```python
env = retro.make('PunchOut-Nes-v0', state=retro.State.NONE, render_mode=None)
obs, info = env.reset()
print(env.get_ram()[913])  # → 96
```

The raw RAM at address 913 showed `96` — the expected starting health value. The ROM was fine. The state file was the problem.

---

## Confirming the RAM Addresses

With the ROM booting, we needed to confirm the `data.json` addresses were still valid for our dump. We used three techniques.

### 1. Monotonically Decreasing Values (Player Health)

Running the game with NOOP actions means the player never dodges — Glass Joe eventually lands punches. Player health should tick down. We recorded 10 RAM snapshots one second apart and looked for addresses that decreased monotonically:

```
addr=913: [52, 41, 41, 30, 30, 19, 19, 8, 0, 0]
addr=914: [52, 41, 41, 30, 30, 19, 19, 8, 8, 0]
```

Address 913 (`health_mac`) and 914 (the "current" mirror) both tracked perfectly with player health draining as Glass Joe landed hits. This matched the `data.json` address exactly.

### 2. Paired Values (Opponent Health)

Both fighters start at full health (`0x60` = 96). We searched for RAM addresses that held value 96 and appeared in pairs — one for each fighter:

```
addr=919 val=96, also at: [920, 921, 922, 926]
```

Address 920 (`health_com`) was in this cluster. Confirmed.

### 3. Hit-Event Differential Analysis

To verify punches could actually damage the opponent, we needed to find which addresses change *specifically when a punch lands*. We recorded frame-by-frame RAM, detected punch-landing events (address 801 briefly going non-zero — the heart/star counter), and checked what else dropped at the same moment:

```python
for t, before, after in hit_events:
    for addr in range(2048):
        d = int(after[addr]) - int(before[addr])
        if -50 < d < 0:
            decreases[addr] = decreases.get(addr, 0) + 1
```

This confirmed address 801 (`heart`) fired correctly on hits, and health addresses responded to damage events.

### Cross-referencing the DataCrystal RAM Map

The [DataCrystal RAM map for Mike Tyson's Punch-Out!!](https://datacrystal.tcrf.net/wiki/Mike_Tyson%27s_Punch-Out!!/RAM_map) provided the authoritative ground truth:

| Hex | Decimal | Description |
|-----|---------|-------------|
| 0x0391 | 913 | Player's next energy level (max 0x60) |
| 0x0392 | 914 | Player's current energy level |
| 0x0398 | 920 | Opponent's next health value |
| 0x0399 | 921 | Opponent's current health value |
| 0x0324 | 804 | Hearts — units digit |
| 0x0300 | 768 | Clock active flag (1 = fight is live) |
| 0x0302 | 770 | Timer — minute digit |
| 0x0305 | 773 | Timer — seconds digit |

Our empirical findings matched the documented map. The original `data.json` addresses were correct for our ROM too.

---

## Building a New State File

With addresses confirmed, the remaining problem was the `Match1.state` file. We needed a new one — saved against our ROM at the moment a fight begins.

The detection condition for "fight just started":

```python
clock_active = int(ram[768])  # 0x0300: clock active
health_mac   = int(ram[913])  # player at full health
health_com   = int(ram[920])  # opponent at full health

if clock_active == 1 and health_mac == 96 and health_com == 96:
    # Fight is live and both fighters are fresh — save state here
```

We booted from `State.NONE`, pressed START every 60 frames to navigate menus, and polled this condition each frame. The fight was detected at frame 1934 (~32 seconds of game time). We then serialized the emulator state and gzip-compressed it — the format stable-retro expects:

```python
import gzip
with gzip.open('Match1.state', 'wb') as f:
    f.write(env.em.get_state())
```

### Validation

Loading the new state file confirmed everything was working:

```
Reset: health_mac=96, health_com=96, clock=1

frame  60: mac=96 com=96 heart=20 score=0
frame 1020: mac=85 com=96 heart=17 score=0   ← Glass Joe lands a hit
frame 1200: mac=74 com=96 heart=14 score=0
...
frame 2160: mac=0  com=96 heart=0  score=0   ← Player KO'd
```

Player health decreasing as Glass Joe attacks, hearts draining over time, KO at zero — all correct. With random actions, we also confirmed the opponent's health could be reduced:

```
frame 241: health_com dropped to 91!
frame 472: health_com dropped to 86!
```

---

## The Final State Space

With the environment verified, we also redesigned the action space to reflect how Punch-Out!! is actually played:

| Index | Action | Buttons |
|-------|--------|---------|
| 0 | NOOP | — |
| 1 | Dodge left | Left |
| 2 | Dodge right | Right |
| 3 | Block / Duck | Down |
| 4 | Right body blow | A |
| 5 | Left body blow | B |
| 6 | Right head punch | Up + A |
| 7 | Left head punch | Up + B |
| 8 | Star uppercut | Start |

Head punches require `Up + A/B` — a combination that wasn't in the original discretizer. The block and duck actions are unified under Down (the distinction is timing, which the agent learns). Star uppercut only fires when the agent has earned stars, so it's naturally rare and high-value.

---

## Summary

| Problem | Solution |
|---------|----------|
| Wrong ROM hash, state file corrupts emulator | Boot from `State.NONE`, bypass state file |
| Unknown if RAM addresses are valid | Empirical: NOOP health drain + hit-event differential |
| Cross-validation of addresses | DataCrystal RAM map confirms addresses |
| No valid fight save state | Auto-detect fight start via clock flag + full health, serialize with gzip |
| Action space didn't match game mechanics | Redesigned with head punches, dodge, block/duck, star uppercut |

The full training setup is in [`train.py`](train.py), with reward shaping in [`envs/wrappers.py`](envs/wrappers.py) and addresses configured in [`config.py`](config.py).
