
from src.wrappers import * 
import random
import numpy as np
import torch

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


env_id = "simglucose/adult1-debug-v0"

# vanilla env , a single OBS Vector
def make_env(seed=SEED, render=False):
    if render:
        base = gym.make(env_id, render_mode="human")
    else:
        base = gym.make(env_id)

    base.reset(seed=seed)
    base.action_space.seed(seed)
    base = ActionclipWrapper(base, low=0.0, high=1.0)
    base = FeatureWrapper(base)
    return base


# stacked-obs env , applies the stacking wrapper, default to 4 obs 
def make_env_stacked(seed=SEED, render=False, k=4):
    if render:
        base = gym.make(env_id, render_mode="human")
    else:
        base = gym.make(env_id)
    base.reset(seed=seed)
    base.action_space.seed(seed)
    base = ActionclipWrapper(base, low=0.0, high=1)
    base = FeatureWrapper(base)
    base = StackObsWrapper(base, k=k)
    return base