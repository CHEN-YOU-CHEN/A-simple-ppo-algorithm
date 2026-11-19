import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from ppo import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(description="載入訓練好的 PPO 模型並評估")
    parser.add_argument("--checkpoint", type=str, required=True, help="train.py 存下來的 .pt 檔")
    parser.add_argument("--episodes", type=int, default=5, help="評估幾局")
    parser.add_argument("--seed", type=int, default=100, help="評估用的隨機種子")
    parser.add_argument("--render", action="store_true", help="開視窗即時顯示遊戲畫面")
    parser.add_argument("--random", action="store_true", help="改用隨機動作，當作比較基準")
    parser.add_argument("--gif", type=str, default=None, help="把第一局錄成 GIF，例如 assets/cartpole.gif")
    parser.add_argument("--fps", type=int, default=30, help="GIF 的播放速度")
    return parser.parse_args()


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ActorCritic(ckpt["state_dim"], ckpt["action_dim"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # 切換成評估模式
    return model, ckpt["env_id"]


def evaluate(env, model, num_episodes=5, seed=None, random_policy=False, record_first=False):
    episode_scores = []
    frames = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=None if seed is None else seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            if record_first and episode == 0:
                frames.append(env.render())

            if random_policy:
                action = env.action_space.sample()  # 隨機選擇動作
            else:
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    # 使用目前 Policy 選擇動作，評估時選擇機率最高的動作
                    probs, _ = model(state_tensor)
                    action = torch.argmax(probs, dim=-1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        episode_scores.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.0f}")

    print("-" * 35)
    print(f"Average Reward: {np.mean(episode_scores):.2f}")
    print(f"Max Reward: {np.max(episode_scores):.2f}")
    print(f"Min Reward: {np.min(episode_scores):.2f}")

    return episode_scores, frames


def save_gif(frames, path, fps=30):
    from PIL import Image

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames[::2]]  # 每兩格取一格，縮小檔案
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps * 2),
        loop=0,
    )
    print(f"GIF 已存到: {path}")


def main():
    args = parse_args()
    model, env_id = load_model(args.checkpoint)

    if args.render:
        render_mode = "human"
    elif args.gif:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = gym.make(env_id, render_mode=render_mode)
    env.action_space.seed(args.seed)

    title = "Random Policy" if args.random else "PPO Evaluation"
    print(f"\n========== {title} ({env_id}) ==========")

    _, frames = evaluate(
        env, model,
        num_episodes=args.episodes,
        seed=args.seed,
        random_policy=args.random,
        record_first=args.gif is not None and render_mode == "rgb_array",
    )
    env.close()

    if args.gif:
        if frames:
            save_gif(frames, args.gif, args.fps)
        else:
            print("錄 GIF 時不能同時開 --render，請拿掉 --render 再試一次")


if __name__ == "__main__":
    main()
