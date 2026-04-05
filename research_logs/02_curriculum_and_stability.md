# Counterpunch: Curriculum Learning, Training Stability, and the Press-Start Rabbit Hole

After ppo_8 cleanly knocked out Glass Joe, the natural next question was: can the agent generalize? Glass Joe is the easiest opponent in the game. If we want something more capable, we need to expose it to harder fights. That led to a significant infrastructure push — new save states, a more robust environment wrapper, and a painful debugging session over a single button press.

---

## Frame Stack: 4 → 8

The first change was bumping `frame_stack` from 4 to 8.

Punch-Out!! is full of opponent tells — brief pre-attack animations that signal which punch is coming. With 4 stacked frames at a 4-frame skip, the agent has a ~267ms window of history. Many of Glass Joe's tells last longer than that, but faster opponents later in the game telegraph their moves on a tighter schedule.

Increasing to 8 frames doubles the temporal window to ~533ms while keeping the CNN architecture identical. The observation space grows from `(4, 84, 84)` to `(8, 84, 84)`. Memory and compute cost increase modestly; the benefit is that the agent can now see a full attack cycle unfold before committing.

---

## Match2: Von Kaiser State Generation

ppo_8 only trained on Glass Joe (`Match1.state`). To push toward harder opponents, we needed a `Match2.state` — the moment Von Kaiser's fight begins.

The naive approach would be to record the game manually and save the emulator state. Instead, `scripts/make_state.py` generates states programmatically. The `find_match1()` function already handled Match1 by booting from power-on and waiting for the fight clock. For Match2, the script needed to play through Glass Joe using a trained model, then capture the state the moment Von Kaiser's fight stabilized.

This turned out to be a multi-round debugging effort.

**Problem 1: wrong detection signal.** The first implementation tried to detect Von Kaiser by `fight_id == 32`. This was wrong — `fight_id` at address 0x0008 stayed at 50 throughout both fights. The correct signal was `match_id` at address 0x0001: 0 for Glass Joe, 1 for Von Kaiser.

**Problem 2: stuck between fights.** After detecting Glass Joe beaten, the script sent NOOP while waiting for Von Kaiser's fight to appear. The game's between-fight screens require pressing START to advance, but pressing START during the between-round break pauses the NES. The fix was a slow START pulse — once per 60 logical steps — that advances screens requiring input without triggering the pause menu during auto-advancing animations.

**Problem 3: Glass Joe health for reliable completion.** The trained model occasionally failed to beat Glass Joe within the timeout, requiring many retry attempts. We cheated Glass Joe's health to 1 at the start of the state so the model reliably finishes the fight.

---

## Curriculum Learning

With Match2 generated, training was updated to use curriculum learning:

```python
state: str = "Match1"
generalization_states: list = ["Match2"]
generalization_prob: float = 0.33
```

On each episode reset, training envs use Match2 (Von Kaiser) 33% of the time. The primary fight remains Glass Joe, which the agent already knows. The Von Kaiser exposure prevents the policy from over-specializing on Glass Joe's patterns and forces it to handle a harder opponent.

---

## ppo13–ppo15: Instability When Scaling Up

When we restarted training with 8-frame stacking, the first several runs failed in instructive ways.

**ppo13**: The run used `ent_coef=0.01`. Policy collapsed early — entropy dropped to 2.195 (near-random), eval reward stuck at -101, value function exploded with `explained_variance=-2.4`. The entropy coefficient was too low for a CNN warming up from scratch with an 8-frame stack. The policy committed to a strategy before the value function could guide it.

**ppo14**: Adjusted to `ent_coef=0.05`, `lr=2.5e-4`. Still unstable. The value function was still oscillating.

**ppo15**: Lowered `lr=2e-4`. Training reward climbed to -36.53 but plateaued — the policy had converged to passive survival (avoiding damage but not attacking). `eval_sticky_prob=0.0` meant all 5 eval episodes were identical (deterministic policy + no stickiness = zero variance), masking how the policy was actually performing. Fixed by setting `eval_sticky_prob=0.05`.

The root diagnostic: all these failures shared premature entropy collapse. `ent_coef=0.05` combined with `lr=2e-4` and `clip_range=0.1` turned out to be the stable combination.

---

## ppo16: Recovery

ppo16 with the corrected config:

```python
learning_rate = 2e-4
ent_coef      = 0.05
clip_range    = 0.1
n_steps       = 256
frame_stack   = 8
sticky_prob   = 0.1
eval_sticky_prob = 0.05
```

By 8 million steps, eval reward had reached +147. The agent was beating Glass Joe reliably and beginning to handle Von Kaiser exposure from the curriculum. Training and eval rewards tracked closely — no sign of the train/eval gap that plagued earlier runs.

---

## Reward Fix: Health Resets After Knockdowns

The reward wrapper had a subtle bug. When an opponent is knocked down, their health resets to a lower value when they get up. The wrapper was computing `opponent_dmg = prev_health_com - health_com`. When `health_com` jumped from 0 to 80 on recovery, this produced `opponent_dmg = -80` — a large negative reward for successfully knocking down the opponent.

The fix was simple: clamp both damage deltas to non-negative values.

```python
opponent_dmg = max(self._prev_health_com - health_com, 0)
player_dmg   = max(self._prev_health_mac - health_mac, 0)
```

Health increases (resets, between-fight reloads) no longer generate spurious rewards or penalties. Knockdown credit comes entirely from the `knockdown_dealt` signal.

---

## The Press-Start Bug

The longest debugging session of the project was over a button press.

After Glass Joe is beaten, the game transitions to `fight_state=0x01` (between fights) and shows a series of screens leading to Von Kaiser's intro. The `KnockdownRecovery` wrapper is responsible for pressing START to advance these screens. Despite multiple attempts, the agent would consistently reach `match_id=1, fight_state=0x01` and then sit frozen for hundreds of steps while the wrapper appeared to send START repeatedly.

The debugging path:

1. **Wrong button combo**: `_START_A` pressed both START and A simultaneously. On inter-fight screens, pressing A may cancel or corrupt the screen transition. Changed to `_START` (START only) — no effect.

2. **Pause/unpause loop**: With `PULSE_ON=3`, three consecutive START frames toggle: press (pause), press (unpause), press (pause). Game stuck in pause cycle. Changed to `PULSE_ON=1` — no effect.

3. **Continuous vs. reset counter**: The original wrapper reset `_inactive_frames=0` on each active-fight frame, causing START to fire immediately on any `fight_state!=0xFF` transition. Replaced with a continuous `_frame` counter that never resets between states — so the first press after transition lands at a random phase, letting auto-advancing animations complete before we accidentally pause them. Still no effect.

4. **Added debug print inside the wrapper**: Confirmed that `[KDR] → SENDING START` was firing every 60 raw frames. The emulator was receiving START. The game wasn't responding.

5. **Root cause**: With `SLOW_PULSE_ON=1` and `_frame` incrementing each KnockdownRecovery call, `StochasticFrameSkip`'s 4-iteration loop sees phases 0, 1, 2, 3 across the 4 calls. Only phase 0 sends START; phases 1-3 send NOOP. So the game received START for exactly **1 raw frame** per cycle. The NES has button debounce logic — a single-frame press doesn't register.

   `make_state.py` (which works) sends START for 4 consecutive frames because the action passes through `StochasticFrameSkip` at the logical-step level, repeating all 4 raw frames. Matching that: `SLOW_PULSE_ON=4` means phases 0, 1, 2, 3 all fire START — all 4 raw frames in the same logical step press the button. The screen advanced immediately.

The fix was one line. Finding it took days.

---

## Current State

ppo16 is the active model, trained to ~8–10M steps with 8-frame stacking and 33% Von Kaiser curriculum. The agent beats Glass Joe reliably and is beginning to land punches on Von Kaiser. The infrastructure (save state generation, KnockdownRecovery, reward shaping, curriculum randomization) is solid enough to support training on further opponents.

Next steps: continue training from the ppo16 checkpoint with the corrected reward and press-start handling, push past Von Kaiser, and eventually generate a Match3 state for Don Flamenco.
