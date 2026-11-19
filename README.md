# PPO from Scratch (PyTorch)

用 PyTorch 從零實作 **Proximal Policy Optimization (PPO)**，並在 Gymnasium `CartPole-v1` 上訓練。
程式碼註解附上每一步對應的公式，適合拿來對照論文學習。

<p align="center">
  <img src="assets/cartpole.gif" width="400" alt="訓練好的 PPO agent 玩 CartPole">
</p>

## 實作內容

- Actor-Critic 共用網路層（policy head + value head）
- **GAE (Generalized Advantage Estimation)**，並正確區分 `terminated` 與 `truncated`：
  被時間上限截斷的那一步會用 `V(s')` bootstrap，真正失敗才把未來價值歸零
- Clipped surrogate objective + value loss + entropy bonus
- Advantage 標準化、梯度裁剪、可選的學習率線性衰減
- 固定 seed，結果可重現

## 專案結構

```
ppo/
  model.py      # ActorCritic 網路
  buffer.py     # 收集軌跡 collect() + compute_gae()
  ppo.py        # ppo_update()：PPO 的 loss 與參數更新
train.py        # 訓練入口，argparse 設定超參數，存 checkpoint 與訓練紀錄
evaluate.py     # 載入模型評估、錄製 GIF、隨機策略對照
```

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方式

```bash
# 訓練（預設 150 輪 × 512 步，CPU 約 1~2 分鐘）
python train.py

# 調整超參數
python train.py --iterations 300 --clip-eps 0.1 --lam 0.9 --anneal-lr --seed 2

# 評估並開視窗看 agent 玩
python evaluate.py --checkpoint checkpoints/ppo_CartPole-v1_seed1.pt --render

# 錄成 GIF
python evaluate.py --checkpoint checkpoints/ppo_CartPole-v1_seed1.pt --gif assets/cartpole.gif

# 隨機策略當作對照組
python evaluate.py --checkpoint checkpoints/ppo_CartPole-v1_seed1.pt --random
```

所有參數可以用 `python train.py --help` 查看。

## 結果

預設超參數、seed = 1，訓練約 77k 步，評估 5 局（選機率最高的動作）：

| Policy | 平均分數 | 最高 | 最低 |
|---|---|---|---|
| Random | 26.6 | 47 | 18 |
| PPO | 293.4 | 500 | 192 |

> CartPole-v1 的滿分為 500。每局分數紀錄存在 `results/*.csv`，可以拿來畫學習曲線。

## 核心公式

**GAE**

$$\delta_t = r_t + \gamma V(s_{t+1})(1 - \text{terminated}_t) - V(s_t)$$

$$A_t = \delta_t + \gamma \lambda A_{t+1} (1 - \text{done}_t)$$

**PPO Clipped Objective**

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

$$L^{CLIP} = \mathbb{E}\left[\min\left(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

**總 Loss**

$$L = -L^{CLIP} + c_1 \cdot \tfrac{1}{2}(R_t - V(s_t))^2 - c_2 \cdot H[\pi_\theta]$$

## 參考資料

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2015
- Huang et al., [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), ICLR Blog Track 2022
