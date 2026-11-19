import torch
import torch.nn as nn
from torch.distributions import Categorical


# 定義 actor-critic 網路
class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()

        # 1. 共用網路層
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 2. policy 網路輸出
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # 3. value 網路輸出 V(s)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # state 為 torch.Tensor，形狀為 (batch_size, state_dim)

        # 1. 共用網路層
        features = self.shared_layers(state)

        # 2. policy 網路輸出
        logits = self.policy_head(features)
        probs = torch.softmax(logits, dim=-1)

        # 3. value 網路輸出 V(s)
        value = self.value_head(features).squeeze(-1)  # 將形狀從 (batch_size, 1) 轉換為 (batch_size,)

        return probs, value

    # 功能一: 取得遊戲採樣(不傳入 action)
    # 功能二: 訓練更新網路(傳入舊的 action)
    def get_action_and_value(self, state, action=None):
        probs, value = self.forward(state)

        dist = Categorical(probs)

        if action is None:
            action = dist.sample()
        else:
            action = torch.as_tensor(action, dtype=torch.int64, device=probs.device)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value
