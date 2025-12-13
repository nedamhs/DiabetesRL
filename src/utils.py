
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
    is_sb3 = hasattr(policy, "predict")

    for t in range(steps):
        if render:
            env.render()

        if is_sb3:
            action, _ = policy.predict(obs, deterministic=deterministic)
        else:
            action = policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        bg = info.get("bg", None)

        rewards.append(float(reward))
        bg_values.append(bg)

        # action can be array-like (Box(1,)) or scalar
        try:
            insulin_list.append(float(np.asarray(action).squeeze()))
        except Exception:
            pass

        if verbose:
            print(f"t={t}, BG={bg}, reward={reward}, action={action}")

        if terminated or truncated:
            break

    # ---- metrics ----
    bg_arr = np.array([b for b in bg_values if b is not None], dtype=float)

    if bg_arr.size > 0:
        TIR = float(np.mean((bg_arr >= 70) & (bg_arr <= 180)) * 100)
        TBR = float(np.mean(bg_arr < 70) * 100)
        TAR = float(np.mean(bg_arr > 180) * 100)
    else:
        TIR = TBR = TAR = float("nan")

    minutes = len(rewards) * 3
    hours = minutes / 60.0

    print("\n--- Summary ---")
    print(f"Steps survived: {len(rewards)}  ({minutes} min, {hours:.2f} hours)\n")
    print(f"Time In Range   (70–180 mg/dL): {TIR:.2f}%")
    print(f"Time Below Range (<70 mg/dL):   {TBR:.2f}%")
    print(f"Time Above Range (>180 mg/dL):  {TAR:.2f}%")

    return {
        "rewards": rewards,
        "bg_values": bg_values,
        "insulin": insulin_list,
        "steps_survived": len(rewards),
        "minutes": minutes,
        "hours": hours,
        "TIR": TIR,
        "TBR": TBR,
        "TAR": TAR,
    }



def eval_and_plot_policy(model, make_env_fn, n_episodes=1):
    """
    Evaluates a PPO policy with deterministic rollouts (480 steps per episode),
    converts normalized CGM to mg/dL, prints action statistics and CGM–action
    correlation, plots CGM vs. action, and returns CGM values, actions,
    and episode-level returns.
    """
    acts, cgms, ep_returns = [], [], []

    for ep in range(n_episodes):
        env = make_env_fn()
        obs, _ = env.reset(seed=SEED + ep)

        total_reward = 0.0
        for _ in range(480):
            action, _ = model.predict(obs, deterministic=True)
            a = float(np.array(action).flatten()[0])

            # obs[0] is normalized CGM → convert to mg/dL
            cgm = float(obs[0] * 400.0)

            acts.append(a)
            cgms.append(cgm)

            obs, reward, term, trunc, _ = env.step(action)
            total_reward += float(reward)
            if term or trunc:
                break

        ep_returns.append(total_reward)

    acts = np.array(acts)
    cgms = np.array(cgms)
    ep_returns = np.array(ep_returns)

    print("Action mean/std/min/max:",
          acts.mean(), acts.std(), acts.min(), acts.max())

    print("corr(CGM, action):",
          np.corrcoef(cgms, acts)[0, 1])

    print("\nEpisode reward mean/std:",
          ep_returns.mean(), ep_returns.std())

    plt.figure(figsize=(8, 4))
    plt.scatter(cgms, acts, s=10, alpha=0.4)
    plt.xlabel("CGM (mg/dL)")
    plt.ylabel("PPO Action (U)")
    plt.title("CGM vs PPO Action")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return cgms, acts, ep_returns


# my custom plots 
def rollout_and_plot_day(model, make_env_fn, max_steps=480):
    """
    Roll out one 24-hour simulation (3-min steps),
    compute BG statistics, and plot BG/CGM, insulin,
    reward, and meals vs time (hours).
    """
    env = make_env_fn()
    obs, info = env.reset(seed=SEED)

    bg_list, cgm_list = [], []
    insulin_list, reward_list, meal_list = [], [], []

    for _ in range(max_steps):
        # CGM from observation (normalized)
        cgm = float(obs[0]) * 400.0
        cgm_list.append(cgm)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # True BG
        bg = info.get("bg", None)
        if bg is None:
            bg = env.env.env.patient.state.BG
        bg_list.append(bg)

        insulin_list.append(float(np.array(action).flatten()[0]))
        reward_list.append(float(reward))
        meal_list.append(float(info.get("meal", 0.0)))

        if terminated or truncated:
            break

    # ---- Stats ----
    bg_arr = np.array(bg_list)
    ins_arr = np.array(insulin_list)
    rew_arr = np.array(reward_list)

    stats = {
        "Mean BG": bg_arr.mean(),
        "Min BG": bg_arr.min(),
        "Max BG": bg_arr.max(),
        "TIR (%)": np.mean((bg_arr >= 70) & (bg_arr <= 180)) * 100,
        "TBR (%)": np.mean(bg_arr < 70) * 100,
        "TAR (%)": np.mean(bg_arr > 180) * 100,
        "Total Insulin": ins_arr.sum(),
        "Total Reward": rew_arr.sum(),
    }

    print("\n===== Simulation Stats =====")
    for k, v in stats.items():
        print(f"{k:20s}: {v:.3f}")

    # ---- Time axis (hours) ----
    time_hr = np.arange(len(bg_list)) * (3 / 60)  # 3 min per step

    fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(time_hr, bg_list, label="True BG")
    axs[0].plot(time_hr, cgm_list, label="CGM", alpha=0.7)
    axs[0].axhline(70, linestyle="--")
    axs[0].axhline(180, linestyle="--")
    axs[0].set_ylabel("Glucose (mg/dL)")
    axs[0].legend()

    axs[1].plot(time_hr, insulin_list)
    axs[1].set_ylabel("Insulin (U/min)")

    axs[2].plot(time_hr, reward_list)
    axs[2].set_ylabel("Reward")

    axs[3].step(time_hr, meal_list, where="post")
    axs[3].set_ylabel("Meal (g CHO)")
    axs[3].set_xlabel("Time (hours)")

    axs[3].set_xticks(np.arange(0, 25, 4))  # 0,4,8,...,24

    plt.tight_layout()
    plt.show()

    return stats


# ======================= functions for evaluating of patient 1 over multiple episodes ==================


def run_episode(policy_type,  env, model=None,const_action=None,  seed=None, max_steps=480):
    """
    policy_type: "ppo", "stacked ppo", "random", or "constant"
    """
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    steps = 0
    done = False

    while (not done) and steps < max_steps:
        if policy_type in ("ppo", "stacked ppo"):
            assert model is not None, f"{policy_type} needs a trained model."
            action, _ = model.predict(obs, deterministic=True)

        elif policy_type == "random":
            action = env.action_space.sample()

        elif policy_type == "constant":
            assert const_action is not None, "constant policy requires const_action"
            action = np.array([const_action], dtype=np.float32)

        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = bool(terminated or truncated)

    return steps



def evaluate_survival(policy_type, make_env_fn, model=None,  n_episodes=20, const_action=None, base_seed=100):
    lengths = []

    if policy_type == "constant":
        title = f"{policy_type.upper()} (CONST={const_action})"
    else:
        title = policy_type.upper()

    for ep in range(n_episodes):
        env = make_env_fn()
        seed = base_seed + ep

        ep_len = run_episode(
            policy_type=policy_type,
            env=env,
            model=model,
            const_action=const_action,
            seed=seed
        )
        lengths.append(ep_len)

    lengths = np.array(lengths)

    mean_steps = lengths.mean()
    mean_minutes = mean_steps * 3
    mean_hours = mean_minutes / 60

    print(f"\n=== Survival results: {title} ===")
    print(f"Episodes: {n_episodes}")
    print(
        f"Mean steps: {mean_steps:.1f} "
        f"({mean_minutes:.1f} min, {mean_hours:.2f} hours)"
    )
    print(f"Std steps:  {lengths.std():.1f}")
    print(f"Min steps:  {lengths.min():.0f}")
    print(f"Max steps:  {lengths.max():.0f}")

    return lengths


# ============================= functions for cross patient generalization test  ==================

def rollout_one_patient( model, patient_name, make_env_fn, max_minutes=24*60,):
    """
    make_env_fn: function(patient_name) -> env
      e.g. lambda p: make_env_stacked_for_patient(p, k=4, render=False)
    """
    env = make_env_fn(patient_name)
    obs, info = env.reset(seed=SEED)

    bg_list = []
    act_list = []

    max_steps = max_minutes // 3  # 3 min per step

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_done = bool(terminated or truncated)

        bg = float(info.get("bg", np.nan))
        bg_list.append(bg)

        act_list.append(float(np.array(action).flatten()[0]))

        if episode_done:
            break

    bg_arr = np.array(bg_list, dtype=float)
    act_arr = np.array(act_list, dtype=float)

    TIR = np.mean((bg_arr >= 70) & (bg_arr <= 180)) * 100
    TBR = np.mean(bg_arr < 70) * 100
    TAR = np.mean(bg_arr > 180) * 100

    stats = {
        "steps": int(len(bg_arr)),
        "hours": float(len(bg_arr) * 3 / 60.0),
        "Mean BG": float(np.mean(bg_arr)) if len(bg_arr) else np.nan,
        "TIR%": float(TIR),
        "TBR%": float(TBR),
        "TAR%": float(TAR),
    }
    return bg_arr, act_arr, stats




def eval_group(model, patient_ids, make_env_fn):
    out = []
    for pid in patient_ids:
        bg, act, stats = rollout_one_patient(model=model, patient_name=pid, make_env_fn=make_env_fn )
        out.append((pid, stats))
    return out




def summarize_group(stats_list, name):
        # stats_list: list of (patient_id, stats_dict)
        tir = np.array([s["TIR%"] for _, s in stats_list], dtype=float)
        tbr = np.array([s["TBR%"] for _, s in stats_list], dtype=float)
        tar = np.array([s["TAR%"] for _, s in stats_list], dtype=float)
        steps = np.array([s["steps"] for _, s in stats_list], dtype=float)
        hours = steps * 3 / 60

        print(f"\n=== {name} (n={len(stats_list)}) ===")
        print(f"Mean survival: {steps.mean():.1f} steps ({hours.mean():.2f} hours)")
        print(f"TIR%: {tir.mean():.2f} ± {tir.std():.2f}")
        print(f"TBR%: {tbr.mean():.2f} ± {tbr.std():.2f}")
        print(f"TAR%: {tar.mean():.2f} ± {tar.std():.2f}")

        return {
            "group": name,
            "n": len(stats_list),
            "mean_steps": steps.mean(),
            "mean_hours": hours.mean(),
            "mean_TIR": tir.mean(),
            "std_TIR": tir.std(),
            "mean_TBR": tbr.mean(),
            "std_TBR": tbr.std(),
            "mean_TAR": tar.mean(),
            "std_TAR": tar.std(),
        }