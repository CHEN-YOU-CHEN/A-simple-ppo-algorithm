import argparse
import csv
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from ppo import ActorCritic, collect, compute_gae, ppo_update


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="使用 PPO 演算法來訓練的 Gymnasium 環境的 cartpole 平衡遊戲")

    # 實驗設定
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Gymnasium 環境名稱")
    parser.add_argument("--seed", type=int, default=1, help="隨機種子，固定後結果可重現")
    parser.add_argument("--iterations", type=int, default=150, help="訓練多少輪 (每輪收集 T 步)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="模型存放資料夾")
    parser.add_argument("--log-dir", type=str, default="results", help="訓練紀錄 (CSV) 存放的資料夾")

    # PPO 超參數
    parser.add_argument("--T", type=int, default=512, help="與環境互動收集的軌跡步數")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子 γ")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE 的 λ")
    parser.add_argument("--epochs", type=int, default=4, help="資料會重複訓練幾次")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch 大小")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip 範圍 ε")
    parser.add_argument("--c1", type=float, default=0.5, help="value loss 係數")
    parser.add_argument("--c2", type=float, default=0.01, help="entropy 係數")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="梯度裁剪上限")
    parser.add_argument("--hidden-dim", type=int, default=64, help="隱藏層大小")

    # 學習率
    parser.add_argument("--lr", type=float, default=7e-4, help="初始學習率")
    parser.add_argument("--anneal-lr", action="store_true", help="開啟學習率線性衰減")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="學習率衰減的下限")

    return parser.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args, on_iteration=None, save=True, verbose=True):
    set_seed(args.seed)

    env = gym.make(args.env_id)
    env.action_space.seed(args.seed)

    # 轉成 Python int: action_space.n 是 numpy 整數，存進 checkpoint 會讓 torch.load(weights_only=True) 載入失敗
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    model = ActorCritic(state_dim, action_dim, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"ppo_{args.env_id}_seed{args.seed}"
    checkpoint_path = os.path.join(args.save_dir, f"{run_name}.pt")
    log_path = os.path.join(args.log_dir, f"{run_name}.csv")

    # 固定環境開局的亂數種子，這樣每次訓練的結果就可以重現
    state, _ = env.reset(seed=args.seed)
    current_score = 0.0
    all_scores = []
    log_rows = []  # (global_step, episode_return)
    start_time = time.time()

    for iteration in range(1, args.iterations + 1):
        # 線性衰減學習率
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / max(args.iterations - 1, 1)
            optimizer.param_groups[0]["lr"] = args.min_lr + frac * (args.lr - args.min_lr)

        # 1. 打遊戲收集軌跡
        buffer, state, current_score, scores = collect(env, model, state, current_score, args.T)
        all_scores.extend(scores)
        global_step = iteration * args.T
        log_rows.extend((global_step, s) for s in scores)

        # 2. 計算 GAE
        buffer = compute_gae(buffer, args.gamma, args.lam)

        # 3. 更新網路
        ppo_update(
            model, optimizer, buffer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            clip_eps=args.clip_eps,
            c1=args.c1,
            c2=args.c2,
            max_grad_norm=args.max_grad_norm,
        )

        if on_iteration is not None:
            on_iteration(iteration, global_step, all_scores)

        if verbose and iteration % 10 == 0 and len(all_scores) > 0:
            avg_score = np.mean(all_scores[-10:])
            print(
                f"Iteration: {iteration:4d} | Steps: {global_step:7d} | "
                f"Avg Score (last 10): {avg_score:6.2f} | Time: {time.time() - start_time:5.1f}s"
            )

    env.close()

    if not save:
        return all_scores

    # 建立存檔的資料夾，exist_ok=True 表示如果資料夾本來就存在不會報錯
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 存模型: 除了權重，也把建網路需要的設定一起存，evaluate.py 才能直接載入
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "env_id": args.env_id,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "args": vars(args),
        },
        checkpoint_path,
    )

    # 存訓練紀錄，之後可以拿來畫學習曲線
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_step", "episode_return"])
        writer.writerows(log_rows)

    print(f"\n模型已存到: {checkpoint_path}")
    print(f"訓練紀錄已存到: {log_path}")
    print(f"評估模型: python evaluate.py --checkpoint {checkpoint_path}")

    return all_scores


def main():
    # 完整讀入命令列的參數
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
