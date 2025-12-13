
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# plots provided by  env.render() of simglucose 
def display_episode(env, policy, steps=480, seed=None, verbose=False, render=True, deterministic=True):
    """
    Run + (optionally) render one episode and print summary metrics.

    Works with:
      1) callable policy:        policy(obs) -> action
      2) SB3 model policy:       policy.predict(obs, deterministic=...) -> (action, state)

    Args
      env: gymnasium env (already wrapped however you want: vanilla or stacked)
      policy: callable or SB3 model (PPO, RecurrentPPO, etc.)
      steps: max steps (480 = 24h if 3 min/step)
      seed: reset seed (if None, don't pass a seed)
      verbose: print per-step lines (BG, reward, action)
      render: call env.render() each step
      deterministic: for SB3 predict(...)

    Returns
      dict with rewards, bg_values, TIR/TBR/TAR, steps_survived, hours, etc.
    """
    # reset (gymnasium style)
    if seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=seed)

    rewards = []
    bg_values = []
    insulin_list = []

    # detect SB3-like model by presence of .predict
    is_sb