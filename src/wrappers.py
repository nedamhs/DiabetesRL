import gymnasium as gym
import numpy as np
from collections import deque


# limit actions from 0-30  to be only 0-1
class ActionclipWrapper(gym.ActionWrapper):
    """
    Restricts the insulin infusion action range.
    SimGlucose uses a default action space of 0–30 U/min, which can be unstable for RL.
    This wrapper limits the action to a smaller continuous range [low, high] (default: 0–1)
    and updates the action_space so PPO trains with the correct bounds.
    """
    def __init__(self, env, low=0.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high
        # the true range
        self.action_space = gym.spaces.Box(low=np.array([low], dtype=np.float32),
                                            high=np.array([high], dtype=np.float32),
                                            dtype=np.float32,
                                             )
    def action(self, act):
        # PPO already outputs in [low, high]; just safe-clip
        return np.clip(act, self.low, self.high)




class FeatureWrapper(gym.Wrapper):
    """
    Build an augmented observation:
       [ normalized_CGM , time_of_day (min), meal_flag, cgm_slope ]

    - CGM is normalized by /400
    - time_of_day is minute_of_day / 1440  in [0, 1]
    - meal_flag = 1 if a meal is present in this step, else 0
    - cgm_slope = Δ(normalized CGM) = CGM_t/400 - CGM_{t-1}/400
    """
    def __init__(self, env):
        super().__init__(env)

        orig_space = env.observation_space
        assert isinstance(orig_space, gym.spaces.Box)

        orig_low  = orig_space.low.astype(np.float32).copy()
        orig_high = orig_space.high.astype(np.float32).copy()

        # CGM normalization bounds (obs[0]) 
        cgm_low = orig_low.copy()
        cgm_high = orig_high.copy()
        cgm_low[0]  = cgm_low[0]  / 400.0
        cgm_high[0] = cgm_high[0] / 400.0

        # time-of-day bounds 
        tod_low  = np.array([0.0], dtype=np.float32)   # 0 = midnight
        tod_high = np.array([1.0], dtype=np.float32)   # 1 = end of day

        #  binary meal flag bounds (0 or 1) 
        mealflag_low  = np.array([0.0], dtype=np.float32)
        mealflag_high = np.array([1.0], dtype=np.float32)

        # CGM slope bounds 
        # slope is Δ(normalized CGM) per step; in practice it's small,
        # but we give a generous range [-2, 2] to be safe.
        cgm_slope_low  = np.array([-2.0], dtype=np.float32)
        cgm_slope_high = np.array([ 2.0], dtype=np.float32)

        #  Final new observation space
        # [ normalized_CGM , time_of_day , meal_flag , cgm_slope ]
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([cgm_low,  tod_low,  mealflag_low,  cgm_slope_low]),
            high=np.concatenate([cgm_high, tod_high, mealflag_high, cgm_slope_high]),
            dtype=np.float32,
            )

        # store previous normalized CGM for slope computation
        self.prev_cgm_norm = None

    # ==========================================================
    # internal helper to build the new feature vector
    # ==========================================================
    def _build_obs(self, obs, info):
        obs = np.array(obs, dtype=np.float32)

        # normalized CGM (obs[0] is mg/dL) 
        raw_cgm = float(obs[0])
        current_cgm_norm = raw_cgm / 400.0

        # CGM slope in normalized space
        if self.prev_cgm_norm is None:
            cgm_slope = 0.0
        else:
            cgm_slope = current_cgm_norm - self.prev_cgm_norm

        # update stored CGM for next step
        self.prev_cgm_norm = current_cgm_norm

        # write back normalized CGM into obs[0]
        obs[0] = current_cgm_norm

        # time-of-day 
        t = info.get("time", None)
        if t is not None:
            minute_of_day = t.hour * 60 + t.minute
            time_of_day = minute_of_day / 1440.0
        else:
            time_of_day = 0.0

        # binary meal flag
        meal_grams = info.get("meal", 0.0)   # grams of CHO
        meal_flag = 1.0 if meal_grams > 0 else 0.0

        return np.concatenate(
            [
                obs,
                np.array([time_of_day], dtype=np.float32),
                np.array([meal_flag], dtype=np.float32),
                np.array([cgm_slope], dtype=np.float32),
            ]
        )
    # ==========================================================
    # override reset / step to access info
    # ==========================================================
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # reset previous CGM so first slope is 0
        self.prev_cgm_norm = None
        feat_obs = self._build_obs(obs, info)
        return feat_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        feat_obs = self._build_obs(obs, info)
        return feat_obs, reward, terminated, truncated, info



class StackObsWrapper(gym.ObservationWrapper):
    """
    Stacks the last k observations to provide short-term temporal context.

    This wrapper maintains a fixed-length buffer (implemented using a deque)
    that stores the most recent k observations. At each timestep, the current
    observation is appended and the oldest is automatically discarded. The
    stacked observation is formed by concatenating the buffered observations
    along the feature dimension, resulting in an observation of size k × d.

    This approach helps mitigate partial observability by allowing the agent
    to infer short-term trends and delayed effects without using a recurrent model.
    """
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.buffer = deque(maxlen=k)

        orig_space = self.observation_space
        assert isinstance(orig_space, gym.spaces.Box), "StackObsWrapper expects a Box observation space"

        orig_low = orig_space.low
        orig_high = orig_space.high

        stacked_low = np.tile(orig_low, k)
        stacked_high = np.tile(orig_high, k)

        self.observation_space = gym.spaces.Box(
            low=stacked_low,
            high=stacked_high,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.array(obs, dtype=np.float32)

        self.buffer.clear()
        for _ in range(self.k):
            self.buffer.append(obs.copy())

        stacked = np.concatenate(list(self.buffer), axis=-1)
        return stacked, info

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)

        if len(self.buffer) == 0:
            for _ in range(self.k):
                self.buffer.append(obs.copy())
        else:
            self.buffer.append(obs.copy())

        stacked = np.concatenate(list(self.buffer), axis=-1)
        return stacked