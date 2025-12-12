
import random
import numpy as np
import torch


SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def display_episode(env, policy_fn, steps=480):
    """
    Runs and displays a full episode using the given policy.
    Used for visualizing baselines (random, zero insulin, etc.).
    
    policy_fn: function mapping obs -> action
               e.g., lambda obs: env.action_space.sample()
    """
    obs, info = env.reset(seed=SEED)
    rewards = []
    bg_values = []

    for t in range(steps):
        env.render()

        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        bg = info.get("bg")
        rewards.append(float(reward))
        bg_values.append(bg)

        if terminated or truncated:
            break

    # Compute metrics
    bg_arr = np.array([b for b in bg_values if b is not None], dtype=float)
    TIR = np.mean((bg_arr >= 70) & (bg_arr <= 180)) * 100
    TBR = np.mean(bg_arr < 70) * 100
    TAR = np.mean(bg_arr > 180) * 100

    # Convert steps to minutes and hours
    minutes = len(rewards) * 3
    hours = minutes / 60.0

    print("\n--- Summary ---")
    print(f"Steps survived: {len(rewards)}  ({minutes} min, {hours:.2f} hours)\n")

    print(f"Time In Range   (70–180 mg/dL): {TIR:.2f}%")
    print(f"Time Below Range (<70 mg/dL):   {TBR:.2f}%")
    print(f"Time Above Range (>180 mg/dL):  {TAR:.2f}%")