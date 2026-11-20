import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # 不開視窗，直接存圖
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

import train as train_module


def parse_args():
    parser = argparse.ArgumentParser(description="用 Optuna 自動找出 PPO 超參數")
    parser.add_argument("--n-trials", type=int, default=30, help="要試幾組超參數")
    parser.add_argument("--timeout", type=int, default=None, help="最多跑幾秒")
    parser.add_argument("--total-steps", type=int, default=76800, help="每次訓練的總步數 (預設 = 150 輪 × 512 步)")
    parser.add_argument("--n-seeds", type=int, default=1, help="每組參數跑幾個 seed 取平均")
    parser.add_argument("--seed", type=int, default=1, help="第一個訓練 seed，同時也是 Optuna sampler 的 seed")
    parser.add_argument("--last-n", type=int, default=20, help="分數 = 最後 N 局的平均分數")
    parser.add_argument("--no-prune", action="store_true", help="關閉 pruning (不提早淘汰表現差的 trial)")
    parser.add_argument("--study-name", type=str, default="ppo_cartpole", help="study 名稱，也是輸出資料夾名稱")
    parser.add_argument("--out-dir", type=str, default="results/optuna", help="輸出資料夾")
    return parser.parse_args()


def score_of(scores, last_n):
    # 評分標準: 最後 N 局的平均分數，還沒打完任何一局就是 0 分
    return float(np.mean(scores[-last_n:])) if scores else 0.0


def suggest_params(trial):
    # 超參數調整範圍
    return {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),     # 學習率差好幾個數量級，用 log 尺度搜尋
        "clip_eps": trial.suggest_float("clip_eps", 0.1, 0.3),
        "gamma": trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995]),
        "lam": trial.suggest_float("lam", 0.8, 1.0),
        "epochs": trial.suggest_int("epochs", 1, 10),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "T": trial.suggest_categorical("T", [128, 256, 512, 1024]),
        "c2": trial.suggest_float("c2", 0.0, 0.05),
        "anneal_lr": trial.suggest_categorical("anneal_lr", [False, True]),
    }


def make_objective(cli):
    base_args = train_module.parse_args([])  # train.py 的預設值

    def objective(trial):
        params = suggest_params(trial)

        seed_scores = []
        for i in range(cli.n_seeds):
            args = argparse.Namespace(**vars(base_args))
            for k, v in params.items():
                setattr(args, k, v)
            args.seed = cli.seed + i
            # T 不同時，每輪收集的步數不同，所以固定「總步數」而不是固定輪數，比較才公平
            args.iterations = max(cli.total_steps // args.T, 1)

            # 只在第一個 seed 回報中間成績，讓 pruner 判斷要不要提早淘汰
            def report(iteration, global_step, all_scores, first=(i == 0)):
                if not first or cli.no_prune or iteration % 10 != 0:
                    return
                trial.report(score_of(all_scores, cli.last_n), step=global_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            scores = train_module.train(args, on_iteration=report, save=False, verbose=False)
            seed_scores.append(score_of(scores, cli.last_n))

        trial.set_user_attr("seed_scores", seed_scores)
        return float(np.mean(seed_scores))

    return objective


def save_plots(study, out_dir):
    from optuna.visualization import matplotlib as ovm

    warnings.filterwarnings("ignore")  # 畫圖時的 experimental / layout 警告不影響結果

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    plots = {
        "optimization_history.png": lambda: ovm.plot_optimization_history(study),
        "param_importances.png": lambda: ovm.plot_param_importances(study),
        "parallel_coordinate.png": lambda: ovm.plot_parallel_coordinate(study),
        "slice.png": lambda: ovm.plot_slice(study),
        "intermediate_values.png": lambda: ovm.plot_intermediate_values(study),
    }
    for name, fn in plots.items():
        if name == "param_importances.png" and len(completed) < 2:
            continue  # 至少要 2 個完成的 trial 才能算重要性
        try:
            ax = fn()
            fig = ax.flat[0].figure if isinstance(ax, np.ndarray) else ax.figure
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, name), dpi=120)
            plt.close(fig)
        except Exception as e:  # 某些圖在 trial 太少時畫不出來，略過就好
            print(f"略過 {name}: {e}")


def main():
    cli = parse_args()
    torch.set_num_threads(1)  # 網路很小，單執行緒反而比較快
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = os.path.join(cli.out_dir, cli.study_name)
    os.makedirs(out_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(out_dir, 'study.db')}"

    # TPE: 根據前面的結果，挑比較有希望的區域繼續試
    # MedianPruner: 訓練到一半，如果表現比之前 trial 同一時間點的中位數還差，就提早停止
    study = optuna.create_study(
        study_name=cli.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cli.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=cli.total_steps // 4),
    )

    def log_trial(study, trial):
        state = trial.state.name
        value = f"{trial.value:6.1f}" if trial.value is not None else "   -  "
        best = study.best_value if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else float("nan")
        print(f"Trial {trial.number:3d} | {state:8s} | score {value} | best {best:6.1f}")

    study.optimize(make_objective(cli), n_trials=cli.n_trials, timeout=cli.timeout, callbacks=[log_trial])

    # 存結果
    study.trials_dataframe().to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    save_plots(study, out_dir)

    best = study.best_trial
    with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({"score": best.value, "trial": best.number, "params": best.params}, f, indent=2)

    # 印出可以直接用最佳參數訓練的指令
    cmd = ["python train.py"]
    for k, v in best.params.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.append(f"{flag} {v:.6g}" if isinstance(v, float) else f"{flag} {v}")
    cmd.append(f"--iterations {max(cli.total_steps // best.params['T'], 1)}")

    print("\n" + "=" * 50)
    print(f"最佳分數: {best.value:.1f} (trial {best.number})")
    for k, v in best.params.items():
        print(f"  {k:12s} = {v}")
    print(f"\n結果存在: {out_dir}")
    print("用最佳參數重新訓練:\n  " + " ".join(cmd))


if __name__ == "__main__":
    main()
