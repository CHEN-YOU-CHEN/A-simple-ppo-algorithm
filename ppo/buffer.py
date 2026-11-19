import numpy as np
import torch


# 收集遊戲軌跡
def collect(env, model, state, current_score=0.0, T=512):
    states = []                     # 狀態
    actions = []                    # 動作
    rewards = []                    # 獎勵
    terminateds = []                # 完成遊戲?
    dones = []                      # 結束?
    log_probs = []                  # 對數機率
    values = []                     # 當前價值
    next_values = []                # 下一個狀態的價值

    episode_scores = []             # 這一輪內打完的每局得分

    for step in range(T):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)  # 將 state 轉換為 tensor 並增加 batch 維度

        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(state_tensor)

        next_state, reward, terminated, truncated, _ = env.step(action.item())

        # 注意: 遊戲結束時 next_state 是最後一個畫面的狀態(不是 reset 後的)，
        # 被 truncated 截斷時要用它的 V(s') 來 bootstrap
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, _, _, next_value = model.get_action_and_value(next_state_tensor)

        done = terminated or truncated

        states.append(state)                    # NumPy 陣列
        actions.append(action.item())           # tensor 轉成 int
        rewards.append(reward)                  # float
        terminateds.append(float(terminated))   # terminated 從 bool 轉成 float，方便計算 GAE
        dones.append(float(done))               # done 從 bool 轉成 float，方便計算 GAE
        log_probs.append(log_prob.item())       # tensor 轉成 float
        values.append(value.item())             # tensor 轉成 float
        next_values.append(next_value.item())   # tensor 轉成 float

        current_score += reward

        # 遊戲結束，重新開一局
        if done:
            episode_scores.append(current_score)
            current_score = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    buffer = {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "terminateds": np.array(terminateds, dtype=np.float32),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "next_values": np.array(next_values, dtype=np.float32),
    }

    return buffer, state, current_score, episode_scores


def compute_gae(buffer, gamma=0.99, lam=0.95):
    # 公式: δ_t = r_t + γ * V(s_{t+1}) * (1 - terminateds[t]) - V(s_t)
    # 解釋: delta = 獎勵 + 折扣後的下一個狀態價值 - 當前狀態價值，但如果這一步就遊戲完成(terminated = 1),就不用管下一個狀態的價值了

    # 公式: A_t = δ_t + γ * λ * A_{t+1} * (1 - dones[t])
    # 解釋: advantages = delta + 折扣後的下一個狀態的 advantages,如果這一步遊戲停止(done = 1),也不用管下一步 advantages 了 

    rewards = buffer["rewards"]
    terminateds = buffer["terminateds"]
    dones = buffer["dones"]
    values = buffer["values"]
    next_values = buffer["next_values"]

    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    # 計算 GAE 由遊戲軌跡最後面往前算
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminateds[t]) - values[t]

        advantages[t] = delta + gamma * lam * last_gae * (1 - dones[t])

        last_gae = advantages[t]

    buffer["advantages"] = advantages
    buffer["returns"] = advantages + values  # GAE + V(s) = R_t
    return buffer
