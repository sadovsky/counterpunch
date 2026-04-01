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

        opponent_dmg     = self._prev_health_com - health_com
        player_dmg       = self._prev_health_mac - health_mac
        heart_delta      = heart - self._prev_heart
        score_delta      = score - self._prev_score
        star_delta       = stars - self._prev_stars
        kd_dealt_delta   = knockdowns_dealt - self._prev_knockdowns_dealt
        kd_taken_delta   = knockdowns_taken - self._prev_knockdowns_taken
        punch_delta      = punches_landed - self._prev_punches_landed

        is_noop = np.all(action == 0)

        shaped_reward = (
            opponent_dmg                    * self.cfg.opponent_damage
            + player_dmg                    * self.cfg.player_damage
            + max(-heart_delta, 0)          * self.cfg.heart_loss
            + score_delta                   * self.cfg.score_weight
            + max(star_delta, 0)            * self.cfg.star_bonus
            + max(kd_dealt_delta, 0)        * self.cfg.knockdown_dealt
            + max(kd_taken_delta, 0)        * self.cfg.knockdown_taken
            + max(punch_delta, 0)           * self.cfg.punch_landed
            + (self.cfg.noop_penalty if is_noop else 0.0)
        )

        # KO bonus: opponent health drops to 0
        if health_com == 0 and self._prev_health_com > 0:
            shaped_reward += self.cfg.ko_bonus

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
    """Presses START whenever the fight clock is inactive.

    Detects knockdowns and between-round pauses via the clock-active flag
    (RAM 768). When the clock stops, Mac is either down (referee count) or
    the round just ended. Holding START gets him back up and advances past
    the inter-round screen as fast as possible.
    """

    ADDR_CLOCK = 768  # 0x0300 — 1 when fight clock is running

    _START = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        ram = self.env.unwrapped.get_ram()
        clock_active = int(ram[self.ADDR_CLOCK]) == 1

        # Clock inactive → knockdown count or between rounds; inject START
        if not clock_active:
            action = self._START

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
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.env.max_episode_steps)
        return env

    return _init
