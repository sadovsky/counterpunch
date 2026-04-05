# Counterpunch: Training a PPO Agent to Play NES Punch-Out!!

I wanted to see if a reinforcement learning agent could learn to throw punches, dodge hooks, and score knockouts in NES Punch-Out!! — starting from nothing but raw pixels. The result is **Counterpunch**: a PPO + CnnPolicy agent built on stable-baselines3 and stable-retro that, after 10 million training steps, consistently knocks out Glass Joe in about 16 seconds of game time. Getting there was messier than I expected.

---

## ROM Setup Hell

The first obstacle had nothing to do with reinforcement learning.

stable-retro ships without ROMs — you have to supply your own legal copy. That's fair. But when I ran `python3 -m stable_retro.import roms/`, it imported exactly zero games. The problem: stable-retro validates ROM files against a list of known SHA1 hashes, and my dump didn't match any of them.

The fix was manual: copy the ROM file into stable-retro's internal data directory and update the `.sha` file with the actual SHA1 of my ROM. Then I discovered that the environment name had to be `PunchOut-Nes-v0`, not `PunchOut-Nes` — a small thing, but it burned some time.

---

## Finding the Right RAM Addresses

NES Punch-Out!! stores game state in RAM, and stable-retro can expose those values as observations or reward signals — but only if you tell it which addresses to watch. The game doesn't come with a pre-configured `data.json`, so I had to find the relevant addresses myself.

I used a three-pronged approach:

1. **Monotonic decrease detection**: ran NOOP episodes and flagged any RAM addresses that only ever decreased. Health values behave this way during a fight.
2. **Hit-event differential analysis**: compared full RAM snapshots before and after landing punches to isolate addresses that changed on contact.
3. **DataCrystal cross-reference**: verified candidates against the community RAM map at datacrystal.tcrf.net.

The addresses I ended up tracking in `data.json`:

| Variable | Address | Notes |
|---|---|---|
| `health_com` | 913 / 920 | Opponent (Glass Joe) health |
| `health_mac` | — | Player (Mac) health |
| `stars` | 834 | Star punch count |
| `knockdowns_dealt` | 977 | Knockdowns landed |
| `knockdowns_taken` | 976 | Knockdowns received |
| `punches_landed` | 945 | Cumulative punch counter |
| `heart` | 835 | Stamina for star punches |
| `score` | — | Tracked, but later zeroed out |

The `score` variable turned out to be stored as BCD (binary-coded decimal), which produced unexpectedly large raw values. I zeroed out the score weight in the reward function rather than deal with the encoding.

---

## Save State Generation

The default `Match1.state` bundled with stable-retro was incompatible with my ROM dump. Rather than debug the mismatch, I wrote `scripts/make_state.py` to generate a fresh save state from scratch.

The script boots from `State.NONE`, advances frames until the fight clock becomes active (RAM address 768 == 1) and both health values are at their starting value of 96, then serializes the emulator state to a gzip-compressed file. The fight first becomes ready at around frame 1934. One early bug: I initially saved the raw bytes without compression, which caused a "Not a gzipped file" error on load. Using `gzip.open()` fixed it.

---

## Designing the Action Space

stable-retro's default action space for NES games is `MultiBinary(9)` — every possible combination of the 9 NES buttons. That's 512 actions, most of which are nonsensical (pressing A+B+Start+Select simultaneously, etc.). A large, mostly-useless action space makes exploration harder.

I replaced it with 9 discrete, semantically meaningful actions:

| ID | Action | Buttons |
|----|--------|---------|
| 0 | NOOP | — |
| 1 | Dodge left | Left |
| 2 | Dodge right | Right |
| 3 | Block / Duck | Down |
| 4 | Right body blow | A |
| 5 | Left body blow | B |
| 6 | Right head punch | Up + A |
| 7 | Left head punch | Up + B |
| 8 | Star uppercut | Start |

The `PunchOutDiscretizer` wrapper maps these 9 actions back to the appropriate button bitmasks. This alone meaningfully improved early training stability — the agent wasn't wasting exploration budget on gibberish inputs.

---

## Reward Shaping

Raw game score makes a poor training signal for RL. I built a `PunchOutRewardWrapper` that computes a shaped reward at each step from the delta values of the RAM variables:

```python
reward = (
    opponent_dmg   *  1.0    # health dealt to opponent
  + player_dmg     * -1.0    # health taken from player
  + knockdown_dealt *  5.0
  + knockdown_taken * -5.0
  + punch_landed   *  0.1
  + star_bonus     *  1.0    # doubled from 0.5
  + ko_bonus       * 10.0    # on KO
  + noop_penalty   * -0.005  # per NOOP step
)
```

The KO bonus of +10 gives the agent a concrete incentive to finish the fight rather than just poke for points. The NOOP penalty discourages passive play. The small punch bonus (+0.1) rewards contact without dominating the signal.

Two reward components I tried and abandoned:

- **Score weight (0.01)**: Redundant with `opponent_dmg`, and the BCD encoding caused unpredictable jumps. Zeroed out.
- **Heart loss penalty (-0.5)**: This one caused the most spectacular failure of the project. More on that below.

---

## Environment Wrappers

The full wrapper stack, in application order:

```
retro.make()
  -> KnockdownRecovery       # presses START when clock inactive (between knockdowns/rounds)
  -> StochasticFrameSkip     # 4-frame skip, sticky action probability
  -> PunchOutRewardWrapper   # custom reward shaping
  -> PunchOutDiscretizer     # 512 -> 9 actions
  -> GrayscaleObservation    # RGB -> grayscale
  -> ResizeObservation       # -> 84x84
  -> FrameStackObservation   # stack 4 frames
  -> TimeLimit               # 4500 steps max (~3 minutes)
```

The `KnockdownRecovery` wrapper handles the moments when the fight clock stops — either during a knockdown or between rounds — by automatically pressing START. Without it, the environment would stall indefinitely waiting for player input.

One subtle bug: `GrayscaleObservation(keep_dim=True)` produces a 4D observation tensor with shape `(4, 84, 84, 1)`. stable-baselines3's `NatureCNN` expects `(C, H, W)` and throws an assertion error on that shape. Setting `keep_dim=False` produced the correct `(4, 84, 84)` stack.

---

## Training Infrastructure

I used `SubprocVecEnv` with 8 parallel environments. Running more than 8 caused deadlocks — 16 envs consistently hung — so I stayed at 8.

**Video recording** was trickier than expected. My first implementation called `model.predict()` inside the recording callback during a rollout. This triggered a CUDA deadlock because the GPU was already mid-computation. The fix was to move recording to `_on_rollout_end()`, which is called between rollouts when the GPU is idle. I record milestone videos at 1%, 25%, 50%, 75%, and 100% of training, saving them to a run-scoped directory under `videos/<timestamp>/`.

**Eval environment**: early runs had eval episodes that never terminated. The environment only marked `done=True` when Mac was knocked out — if the agent was losing, the episode ran forever. Adding `TimeLimit(4500)` to the eval env fixed this. I also had to switch the eval env from `DummyVecEnv` (which runs in the main process) to `SubprocVecEnv` to avoid "multiple emulator instances" errors.

---

## The ppo_7 Collapse

This is the part I want to spend some time on, because it's the most instructive thing that happened.

After ppo_6 showed a persistent train/eval gap — the agent was learning timing that depended on sticky actions (buttons staying pressed from previous steps) but eval ran with no stickiness — I sat down to tune. I lowered `sticky_prob` from 0.25 to 0.1 to close the gap. I also decided to add a penalty for losing heart stamina, reasoning that it would teach the agent to preserve its star punch meter.

The `heart` RAM address seemed reasonable: address 835, values I assumed to be in the 0–96 range like everything else. I added `heart_loss = -0.5`.

The first rollout of ppo_7 came back with a mean reward of **-4014** per episode.

The heart value turned out to operate on a completely different scale. I don't know the exact encoding, but the per-episode accumulation of heart penalties dwarfed every other signal in the reward function by several orders of magnitude. The agent had no useful gradient to follow — just an enormous negative signal it couldn't escape.

By step 35,000, entropy had collapsed to 0. The policy had become fully deterministic, spamming a single action. Rewards continued to slide toward -5000 and beyond.

The fixes: zero out `heart_loss`, restore KO-triggered episode termination (which had been accidentally dropped), and raise `ent_coef` from 0.02 to 0.05 to counteract the tendency toward premature convergence. I started ppo_8 fresh.

The lesson: when adding a new reward component, always sanity-check the actual scale of the underlying variable before multiplying by anything. One unvetted penalty destroyed 35,000 steps of training in seconds.

---

## ppo_8: The Successful Run

ppo_8 ran for 10 million steps with the corrected reward function and updated hyperparameters:

```python
learning_rate = 2.5e-4  # with linear decay to 0
n_steps       = 128
batch_size    = 256
n_epochs      = 4
gamma         = 0.99
gae_lambda    = 0.95
clip_range    = 0.1
ent_coef      = 0.05
frame_skip    = 4
sticky_prob   = 0.1
```

Training progression:

- **Steps 0–500k**: Rollout reward climbed from ~87 to the low 90s. The agent was learning to land punches but not yet finishing fights consistently.
- **Steps 500k–2M**: Rollout reward climbed from ~90 to 112–113. Episode length dropped from ~520 steps to ~240 steps. The agent was learning to KO Glass Joe quickly.
- **Steps 2M–10M**: Rollout reward held steady at 112–113. Episode length stable at ~240 steps. Eval reward, noisy until 2M, converged to a consistent +112–114.

Explained variance reached 0.998 by the end of training — the value function was predicting returns with near-perfect accuracy. Entropy held steady around -1.86 throughout; the raised `ent_coef` prevented the kind of collapse that killed ppo_7.

The policy effectively converged around 2 million steps. The remaining 8 million steps didn't hurt, but they didn't add much either. In retrospect, an earlier stopping criterion would have saved wall-clock time.

---

## Results

The ppo_8 agent:

- Consistently knocks out Glass Joe
- Does it in ~240 steps x 4 frame skip = ~16 seconds of game time
- Achieves a win rate of approximately 95%+ from 2 million steps onward
- Rollout reward stabilizes at 112–113 (decomposed roughly as: KO bonus 10 + health damage 96 + punch bonuses)

The agent developed a specific style: rapid body blows, occasional head punches, and very little dodging. Glass Joe has minimal counter-aggression, so the optimal strategy is essentially pure offense — which is what the agent found.

---

## What I Would Do Differently

**Validate reward component scales before training.** The ppo_7 collapse was entirely preventable. A five-line script that prints the per-step delta for every RAM variable would have caught the heart encoding issue immediately.

**Add earlier stopping.** ppo_8's policy converged at 2M steps. Eight million additional steps were redundant. Training a harder opponent (Piston Honda, Bald Bull) would have been a better use of that compute.

**Increase curriculum difficulty.** Glass Joe is the easiest opponent in the game. The architecture is sound for harder fights, but the reward shaping and action timing would need to be re-evaluated. Star punches, for instance, become much more important against later opponents.

**Instrument RAM variables first.** I spent a non-trivial amount of time on ROM import issues and save state generation before writing a single line of RL code. Setting up proper tooling to inspect and validate the game environment before touching the agent would have saved time overall.

---

## Code

The full project — environment wrappers, reward shaping, training script, and state generation — is in this repository under `scripts/` and the main training entry point. The final model weights are saved under `models/`.