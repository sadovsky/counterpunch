"""Custom environment wrappers for Punch-Out!! NES."""

import random

import gymnasium as gym
import numpy as np
import stable_retro as retro

from config import RewardConfig


# NES controller button order for stable-retro:
# B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A
# Indices: 0    1      2       3     4    5      6     7     8
BUTTON_NAMES = ["B", "NULL", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]

# Filtered action table: each row is a MultiBinary(9) NES button press
ACTION_TABLE = [
    #  B  NL SEL STA UP  DN  LT  RT  A
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: NOOP
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 1: Dodge left
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 2: Dodge right
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 3: Block/duck (down)
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 4: Right body blow (A)
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 5: Left body blow (B)
    [0, 0, 0, 0, 1, 0, 0, 0, 1],  # 6: Right head punch (up + A)
    [1, 0, 0, 0, 1, 0, 0, 0, 0],  # 7: Left head punch (up + B)
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 8: Star uppercut (start)
]


class PunchOutDiscretizer(gym.ActionWrapper):
    """Maps a small set of discrete actions to meaningful Punch-Out button combos."""

    def __init__(self, env):
        super().__init__(env)
        self._actions = np.array(ACTION_TABLE, dtype=np.int8)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action].copy()


class PunchOutRewardWrapper(gym.Wrapper):
    """Custom reward shaping using Punch-Out RAM values.

    Reads info dict from stable-retro containing health_com, health_mac,
    heart, and score. Computes a shaped reward based on deltas.
    """

    def __init__(self, env, reward_config: RewardConfig | None = None):
        super().__init__(env)
        self.cfg = reward_config or RewardConfig()
        self._prev_health_com = 0
        self._prev_health_mac = 0
        self._prev_heart = 0
        self._prev_score = 0
        self._prev_stars = 0
        self._prev_knockdowns_dealt = 0
        self._prev_knockdowns_taken = 0
        self._prev_punches_landed = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_health_com = info.get("health_com", 0)
        self._prev_health_mac = info.get("health_mac", 0)
        self._prev_heart = info.get("heart", 0)
        self._prev_score = info.get("score", 0)
        self._prev_stars = info.get("stars", 0)
        self._prev_knockdowns_dealt = info.get("knockdowns_dealt", 0)
        self._prev_knockdowns_taken = info.get("knockdowns_taken", 0)
        self._prev_punches_landed = info.get("punches_landed", 0)
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        health_com       = info.get("health_com", self._prev_health_com)
        health_mac       = info.get("health_mac", self._prev_health_mac)
        heart            = info.get("heart", self._prev_heart)
        score            = info.get("score", self._prev_score)
        stars            = info.get("stars", self._prev_stars)
        knockdowns_dealt = info.get("knockdowns_dealt", self._prev_knockdowns_dealt)
        knockdowns_taken = info.get("knockdowns_taken", self._prev_knockdowns_taken)
        punches_landed   = info.get("punches_landed", self._prev_punches_landed)

        # Clamp to [0, ...): health resets after knockdowns should not generate
        # reward or penalty — those are accounted for by knockdown_dealt/taken.
        opponent_dmg     = max(self._prev_health_com - health_com, 0)
        player_dmg       = max(self._prev_health_mac - health_mac, 0)
        heart_delta      = heart - self._prev_heart
        score_delta      = score - self._prev_score
        star_delta       = stars - self._prev_stars
        kd_dealt_delta   = knockdowns_dealt - self._prev_knockdowns_dealt
        kd_taken_delta   = knockdowns_taken - self._prev_knockdowns_taken
        punch_delta      = punches_landed - self._prev_punches_landed

        stars_used       = max(-star_delta, 0)   # positive when a star is consumed
        star_landed      = 1 if (stars_used > 0 and opponent_dmg > 0) else 0

        is_noop = np.all(action == 0)

        shaped_reward = (
            opponent_dmg                    * self.cfg.opponent_damage
            + player_dmg                    * self.cfg.player_damage
            + max(-heart_delta, 0)          * self.cfg.heart_loss
            + score_delta                   * self.cfg.score_weight
            + max(star_delta, 0)            * self.cfg.star_bonus
            + stars_used                    * self.cfg.star_used
            + star_landed                   * self.cfg.star_hit
            + max(kd_dealt_delta, 0)        * self.cfg.knockdown_dealt
            + max(kd_taken_delta, 0)        * self.cfg.knockdown_taken
            + max(punch_delta, 0)           * self.cfg.punch_landed
            + (self.cfg.noop_penalty if is_noop else 0.0)
        )

        # KO bonus: opponent health drops to 0
        if health_com == 0 and self._prev_health_com > 0:
            shaped_reward += self.cfg.ko_bonus

        # Terminate when Mac is KO'd during an active fight.
        # Guard health_com > 0 to avoid false triggers during post-KO
        # transitions where RAM health values are temporarily unreliable.
        if health_mac == 0 and self._prev_health_mac > 0 and health_com > 0:
            terminated = True

        self._prev_health_com       = health_com
        self._prev_health_mac       = health_mac
        self._prev_heart            = heart
        self._prev_score            = score
        self._prev_stars            = stars
        self._prev_knockdowns_dealt = knockdowns_dealt
        self._prev_knockdowns_taken = knockdowns_taken
        self._prev_punches_landed   = punches_landed

        return obs, shaped_reward, terminated, truncated, info


class KnockdownRecovery(gym.Wrapper):
    """Handles knockdowns and between-round/fight "press start" screens.

    Two distinct cases:
    - clock==0, fight==0xFF: Mac knocked down mid-fight. Pulse START+A fast
      (3 on / 12 off) to fill the get-up meter.
    - clock==1, fight!=0xFF: "Press start" screen between rounds or fights.
      Pulse START slowly (3 on / 57 off, ~1 press/sec) to advance the screen
      without accidentally pausing an active fight.
    """

    ADDR_CLOCK       = 768   # 0x0300 — 1 when fight clock is running
    ADDR_FIGHT_STATE = 4     # 0x0004 — 0xFF=active fight, 0x01=between rounds
    FIGHT_ACTIVE     = 0xFF

    # Knockdown recovery: fast multi-frame pulse to fill the get-up meter.
    FAST_PULSE_ON    = 3
    FAST_CYCLE       = 15
    # Press-start screens: single-frame press on a slow cycle.
    # Single frame avoids the pause→unpause→pause trap that 3 held frames cause.
    # Continuous _frame counter (not reset on fight_state transition) means the
    # first press after transition lands at a random phase, letting auto-advancing
    # animations play out before we accidentally pause them.
    SLOW_PULSE_ON    = 4   # hold 4 frames to clear NES debounce (matches make_state.py)
    SLOW_CYCLE       = 60

    _START_A = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.int8)  # knockdown recovery
    _START   = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)  # press-start screens
    _NOOP    = np.zeros(9, dtype=np.int8)

    def __init__(self, env):
        super().__init__(env)
        self._frame = 0

    def reset(self, **kwargs):
        self._frame = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        ram = self.env.unwrapped.get_ram()
        clock_active = int(ram[self.ADDR_CLOCK]) == 1
        fight_state  = int(ram[self.ADDR_FIGHT_STATE])

        if not clock_active and fight_state == self.FIGHT_ACTIVE:
            # Mac knocked down mid-fight — fast START+A pulse
            phase = self._frame % self.FAST_CYCLE
            action = self._START_A if phase < self.FAST_PULSE_ON else self._NOOP
        elif fight_state != self.FIGHT_ACTIVE:
            # Between rounds, post-KO, or press-start screen — slow single-frame START pulse.
            phase = self._frame % self.SLOW_CYCLE
            action = self._START if phase < self.SLOW_PULSE_ON else self._NOOP

        self._frame += 1
        return self.env.step(action)


class StochasticFrameSkip(gym.Wrapper):
    """Skip frames with a sticky-action probability.

    With probability `sticky_prob`, the previous action is repeated instead
    of the current one. This prevents the agent from relying on precise
    frame-level timing that wouldn't transfer.
    """

    def __init__(self, env, n_frames: int = 4, sticky_prob: float = 0.25):
        super().__init__(env)
        self.n_frames = n_frames
        self.sticky_prob = sticky_prob
        self._prev_action = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.n_frames):
            if self._prev_action is not None and np.random.random() < self.sticky_prob:
                use_action = self._prev_action
            else:
                use_action = action

            obs, reward, terminated, truncated, info = self.env.step(use_action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        self._prev_action = action
        return obs, total_reward, terminated, truncated, info


def make_env(config, render_mode=None, eval_env=False):
    """Factory function returning a callable that creates a wrapped Punch-Out env.

    Returns a function compatible with stable-baselines3 make_vec_env.
    """

    def _init():
        state = config.env.state
        if (
            not eval_env
            and config.env.generalization_states
            and random.random() < config.env.generalization_prob
        ):
            state = random.choice(config.env.generalization_states)

        env = retro.make(
            game=config.env.game,
            state=state,
            render_mode=render_mode,
        )
        env = KnockdownRecovery(env)
        sticky_prob = config.env.eval_sticky_prob if eval_env else config.env.sticky_prob
        env = StochasticFrameSkip(
            env,
            n_frames=config.env.frame_skip,
            sticky_prob=sticky_prob,
        )
        env = PunchOutRewardWrapper(env, config.reward)
        env = PunchOutDiscretizer(env)
        if config.env.grayscale:
            env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
        env = gym.wrappers.ResizeObservation(env, shape=config.env.resize)
        env = gym.wrappers.FrameStackObservation(env, stack_size=config.env.frame_stack)
        max_steps = config.env.eval_max_episode_steps if eval_env else config.env.max_episode_steps
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        return env

    return _init
