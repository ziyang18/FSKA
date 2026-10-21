from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FSKA_code.models import baseTransformer


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    args = SimpleNamespace()
    args.random_seed = cli_args.seed
    args.N_EPOCHS = cli_args.epochs
    args.LEARNING_RATE = cli_args.lr
    args.HID_DIM = 64
    args.KEY_DIM = 64
    args.ENC_LAYERS = 1
    args.ENC_HEADS = 8
    args.ENC_DROPOUT = 0.0
    args.M = 64
    args.M_frska = 96
    args.prior_var = 0.1
    args.kl_lambda = 0.1
    args.kl_lambda_frska = 0.001
    args.copula_lambda = 1.0
    args.p_norm = 2.0
    args.att_dropout = 0.1
    args.device = cli_args.device
    args.eval_batch_size = cli_args.eval_batch_size
    args.frska_extra_epochs = cli_args.frska_extra_epochs
    args.frska_positive_map = True
    args.dot_qk_mode = "normal"
    args.n_points = cli_args.n_points
    args.frska_num_flows = cli_args.frska_num_flows
    args.frska_num_mixtures = cli_args.frska_num_mixtures
    return args


def experiment_specs():
    return [
        ("Transformer", "dot", False, False, False, "normal", "learnable", {}),
        (
            "FSKA",
            "frska",
            False,
            False,
            False,
            "no_qk",
            "fixed_orth",
            {"kl_lambda_frska": 0.001, "M_frska": 96, "frska_positive_map": True},
        ),
    ]


def generate_nonstationary_data(n_points=500, train_ratio=0.7, random_seed=42, device="cpu", split_mode="random"):
    rng = np.random.default_rng(random_seed)
    t = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    x_aux = rng.normal(loc=0.0, scale=0.7, size=n_points).astype(np.float32)

    s1 = 1.0 / (1.0 + np.exp(-12.0 * (t - 0.26)))
    s2 = 1.0 / (1.0 + np.exp(-12.0 * (t - 0.56)))
    s3 = 1.0 / (1.0 + np.exp(-12.0 * (t - 0.82)))
    w0 = np.clip(1.0 - s1, 0.0, 1.0)
    w1 = np.clip(s1 - s2, 0.0, 1.0)
    w2 = np.clip(s2 - s3, 0.0, 1.0)
    w3 = np.clip(s3, 0.0, 1.0)

    phase = 2.0 * np.pi * (0.55 * t + 0.95 * t * t + 0.35 * t * t * t)
    amp = 0.50 + 0.12 * np.sin(2.0 * np.pi * (0.28 * t + 0.15 * t * t))
    regime_signal = (
        w0 * (0.30 * np.sin(1.2 * np.pi * t + 0.20 * x_aux))
        + w1 * (0.34 * np.cos(1.9 * np.pi * t - 0.16 * x_aux))
        + w2 * (0.32 * np.sin(2.6 * np.pi * t + 0.16 * x_aux))
        + w3 * (0.36 * np.cos(3.2 * np.pi * t - 0.14 * x_aux))
    )
    mean_shift = -0.18 * w0 + 0.06 * w1 + 0.20 * w2 - 0.07 * w3
    trend = 0.22 * (t - 0.5) ** 3 + 0.06 * t
    hetero_std = 0.012 + 0.010 * (1.0 + np.sin(2.0 * np.pi * 0.60 * t)) + 0.022 * w2 + 0.032 * w3
    noise = rng.normal(loc=0.0, scale=hetero_std, size=n_points).astype(np.float32)

    y_clean = amp * np.sin(phase + 0.10 * x_aux) + regime_signal + trend + mean_shift
    y = (y_clean + noise).reshape(-1, 1).astype(np.float32)
    x = np.stack(
        [t, np.sin(2.0 * np.pi * t), np.cos(2.0 * np.pi * t), t**2, t**3, x_aux, w1, w2, w3, hetero_std],
        axis=1,
    ).astype(np.float32)

    if split_mode == "time":
        n_train = int(n_points * train_ratio)
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_points)
    else:
        all_idx = np.arange(n_points)
        rng.shuffle(all_idx)
        n_train = int(n_points * train_ratio)
        train_idx = np.sort(all_idx[:n_train])
        test_idx = np.sort(all_idx[n_train:])

    x_train = torch.tensor(x[train_idx], device=device)
    y_train = torch.tensor(y[train_idx], device=device)
    x_test = torch.tensor(x[test_idx], device=device)
    y_test = torch.tensor(y[test_idx], device=device)
    tr_mask = torch.ones(len(train_idx), 1, dtype=torch.float32, device=device)
    te_mask = torch.ones(len(test_idx), 1, dtype=torch.float32, device=device)

    tr_data = (x_train, y_train, tr_mask, train_idx)
    te_data = (x_test, y_test, te_mask, test_idx)
    info = {
        "split_mode": split_mode,
        "input_dim": x.shape[1],
        "output_dim": 1,
        "n_points": int(n_points),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "time_axis": t,
        "target_full": y[:, 0],
        "target_clean_full": y_clean.astype(np.float32),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    return tr_data, te_data, info


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _safe_kld(kld):
    if torch.isnan(kld) or torch.isinf(kld):
        return torch.tensor(0.0, device=kld.device)
    return torch.clamp(kld, min=0.0, max=1000.0)


def _reg_metrics(y_true, y_pred):
    mse = F.mse_loss(y_pred, y_true).item()
    mae = F.l1_loss(y_pred, y_true).item()
    y_mean = torch.mean(y_true)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_mean) ** 2) + 1e-8
    r2 = (1.0 - ss_res / ss_tot).item()
    return mse, mae, r2


def train_regression(model, data, optimizer, kl_weight):
    model.train()
    x, y, mask, _ = data
    optimizer.zero_grad()
    output, kld = model(x)
    pred = output.squeeze(0).squeeze(-1)
    target = y.squeeze(-1)
    mse = F.mse_loss(pred, target)
    loss = mse + kl_weight * _safe_kld(kld) / torch.sum(mask)
    if torch.isnan(loss) or torch.isinf(loss):
        return 0.0, 0.0, 0.0, 0.0
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    mse_val, mae_val, r2_val = _reg_metrics(target.detach(), pred.detach())
    return loss.item(), mse_val, mae_val, r2_val


def evaluate_regression(model, data, eval_batch_size=1):
    model.eval()
    with torch.no_grad():
        x, y, _, _ = data
        target = y.squeeze(-1)
        if eval_batch_size <= 0:
            output, _ = model(x)
            pred = output.squeeze(0).squeeze(-1)
        else:
            preds = []
            for start in range(0, x.shape[0], eval_batch_size):
                end = min(start + eval_batch_size, x.shape[0])
                batch_out, _ = model(x[start:end])
                preds.append(batch_out.squeeze(0).squeeze(-1))
            pred = torch.cat(preds, dim=0)
        return pred, *_reg_metrics(target, pred)


def run_experiment(exp_name, att_type, args, tr_data, te_data, num_data, input_dim, output_dim, device, freeze_qk, freeze_v, shared_flow, qk_mode, v_mode):
    args.att_type = att_type
    if att_type == "frska":
        args.frska_shared_flow = shared_flow
        args.frska_qk_mode = qk_mode
        args.frska_v_mode = v_mode

    original_m = args.M
    if att_type == "frska":
        args.M = args.M_frska

    model = baseTransformer(
        args,
        num_data,
        input_dim,
        output_dim,
        args.HID_DIM,
        args.ENC_LAYERS,
        args.ENC_HEADS,
        args.HID_DIM,
        args.ENC_DROPOUT,
        device,
        freeze_qk=freeze_qk,
        freeze_v=freeze_v,
    ).to(device)

    if att_type == "frska":
        args.M = original_m
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE * 0.8, weight_decay=5e-5)
        kl_weight = getattr(args, "kl_lambda_frska", 0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
        kl_weight = getattr(args, "kl_lambda", 0.1)

    num_params = count_parameters(model)
    best_test_mse = float("inf")
    best_epoch = 0
    best_pred = None
    best_te_mae = float("inf")
    best_te_r2 = -float("inf")
    train_mse_list = []
    test_mse_list = []
    n_epochs = args.N_EPOCHS + (args.frska_extra_epochs if att_type == "frska" else 0)
    start_time = time.time()

    epoch_iter = range(n_epochs)
    if HAS_TQDM:
        epoch_iter = tqdm(epoch_iter, desc=exp_name, ncols=100, leave=False)

    for epoch in epoch_iter:
        _, tr_mse, _, _ = train_regression(model, tr_data, optimizer, kl_weight)
        te_pred, te_mse, te_mae, te_r2 = evaluate_regression(model, te_data, args.eval_batch_size)
        train_mse_list.append(tr_mse)
        test_mse_list.append(te_mse)
        if te_mse < best_test_mse:
            best_test_mse = te_mse
            best_epoch = epoch + 1
            best_pred = te_pred.detach().cpu().numpy()
            best_te_mae = te_mae
            best_te_r2 = te_r2
        if HAS_TQDM:
            epoch_iter.set_postfix(best=f"{best_test_mse:.4f}", te=f"{te_mse:.4f}")

    return {
        "name": exp_name,
        "att_type": att_type,
        "best_test_mse": float(best_test_mse),
        "best_test_mae": float(best_te_mae),
        "best_test_r2": float(best_te_r2),
        "best_epoch": int(best_epoch),
        "runtime": float(time.time() - start_time),
        "num_params": int(num_params),
        "predictions": best_pred.reshape(-1, 1) if best_pred is not None else np.zeros((te_data[0].shape[0], 1), dtype=np.float32),
        "te_idx": te_data[3],
        "train_mse_list": train_mse_list,
        "test_mse_list": test_mse_list,
        "pred_std": np.zeros((te_data[0].shape[0], 1), dtype=np.float32),
    }


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def plot_prediction_single_model(result, data_info, out_path):
    t = np.asarray(data_info["time_axis"])
    y_true = np.asarray(data_info["target_full"])
    train_idx = np.asarray(data_info["train_idx"])
    test_idx = np.asarray(data_info["test_idx"])
    pred = np.asarray(result["predictions"]).reshape(-1)
    test_t = t[test_idx]
    order = np.argsort(test_t)

    plt.figure(figsize=(10, 5))
    plt.plot(t, y_true, color="#9ca3af", linewidth=2.2, alpha=0.95, zorder=1, label="GT")
    plt.scatter(t[train_idx], y_true[train_idx], s=9, color="#9ca3af", alpha=0.20, zorder=0)
    plt.plot(test_t[order], pred[order], color="#2563eb", linewidth=2.0, alpha=0.95, zorder=3, label="Pred")
    plt.title(
        f"{result['name']} | MSE={result['best_test_mse']:.4f}, MAE={result['best_test_mae']:.4f}, R2={result['best_test_r2']:.4f}",
        fontweight="bold",
    )
    plt.xlabel("Normalized time", fontweight="bold")
    plt.ylabel("Target value", fontweight="bold")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_prediction_grid(results, data_info, out_path):
    t = np.asarray(data_info["time_axis"])
    y_true = np.asarray(data_info["target_full"])
    test_idx = np.asarray(data_info["test_idx"])
    fig, axes = plt.subplots(1, len(results), figsize=(5.4 * len(results), 3.8), squeeze=False)
    for i, result in enumerate(results):
        ax = axes[0][i]
        pred = np.asarray(result["predictions"]).reshape(-1)
        test_t = t[test_idx]
        order = np.argsort(test_t)
        ax.plot(t, y_true, color="#9ca3af", linewidth=1.8, alpha=0.95, zorder=1)
        ax.plot(test_t[order], pred[order], color="#2563eb", linewidth=1.8, alpha=0.95, zorder=4)
        ax.set_title(f"{result['name']}\nMSE={result['best_test_mse']:.4f}", fontweight="bold")
        ax.grid(alpha=0.2)
        ax.set_xlabel("t", fontweight="bold")
        ax.set_ylabel("y", fontweight="bold")
    legend_handles = [
        Line2D([0], [0], color="#9ca3af", lw=2.0, label="GT"),
        Line2D([0], [0], color="#2563eb", lw=2.0, label="Pred"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_mse_bar(results, out_path):
    names = [r["name"] for r in results]
    mses = [float(r["best_test_mse"]) for r in results]
    plt.figure(figsize=(6, 4.5))
    bars = plt.bar(names, mses, color=["#2563eb", "#059669"])
    plt.ylabel("Best Test MSE", fontweight="bold")
    plt.title("Toy Non-stationary Regression", fontweight="bold")
    plt.grid(axis="y", alpha=0.25)
    for bar, mse in zip(bars, mses):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{mse:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_report(path, settings, results):
    lines = [
        "=" * 96,
        "FSKA Dual-Model Toy Non-stationary Regression Report",
        "=" * 96,
        "",
        "Run setup:",
        f"  seed: {settings['seed']}",
        f"  n_points: {settings['n_points']}",
        f"  train/test: {settings['n_train']}/{settings['n_test']} ({settings['split_mode']} split)",
        f"  epochs: {settings['epochs']}",
        f"  frska_extra_epochs: {settings['frska_extra_epochs']}",
        f"  learning_rate: {settings['lr']}",
        f"  eval_batch_size: {settings['eval_batch_size']}",
        f"  device: {settings['device']}",
        "",
        "-" * 96,
        f"{'Model':<16} {'Best MSE':<12} {'Best MAE':<12} {'Best R2':<10} {'Best Epoch':<12}",
        "-" * 96,
    ]
    for result in results:
        lines.append(
            f"{result['name']:<16} {result['best_test_mse']:<12.6f} {result['best_test_mae']:<12.6f} "
            f"{result['best_test_r2']:<10.4f} {result['best_epoch']:<12}"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- FSKA is the paper's OURS method.",
            "- FSKA here uses independent kernels across heads (shared_flow=False).",
            "- Toy data generation and output format follow the existing nonstationary V2 experiment.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_outputs(out_dir: Path, settings: dict, results: list[dict], data_info: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results_seed.json", "w", encoding="utf-8") as f:
        json.dump({"settings": _to_serializable(settings), "results": _to_serializable(results)}, f, ensure_ascii=False, indent=2)
    write_report(str(out_dir / "ablation_report.txt"), settings, results)
    np.savez_compressed(
        out_dir / "plot_repro_data.npz",
        time_axis=np.asarray(data_info["time_axis"]),
        target_full=np.asarray(data_info["target_full"]),
        target_clean_full=np.asarray(data_info["target_clean_full"]),
        train_idx=np.asarray(data_info["train_idx"]),
        test_idx=np.asarray(data_info["test_idx"]),
    )
    per_model_dir = out_dir / "per_model"
    single_plot_dir = out_dir / "plots_single"
    per_model_dir.mkdir(exist_ok=True)
    single_plot_dir.mkdir(exist_ok=True)
    for index, result in enumerate(results, start=1):
        with open(per_model_dir / f"{index:02d}_{result['name']}.json", "w", encoding="utf-8") as f:
            json.dump(_to_serializable(result), f, ensure_ascii=False, indent=2)
        plot_prediction_single_model(result, data_info, single_plot_dir / f"{index:02d}_{result['name']}.png")
    plot_prediction_grid(results, data_info, out_dir / "prediction_grid_all_models.png")
    plot_mse_bar(results, out_dir / "mse_bar.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_points", type=int, default=500)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--split_mode", type=str, default="random", choices=["time", "random"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--frska_extra_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--frska_num_flows", type=int, default=3)
    parser.add_argument("--frska_num_mixtures", type=int, default=4)
    parser.add_argument("--only_experiment", type=str, default=None)
    cli_args = parser.parse_args()

    device = resolve_device(cli_args.device)
    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)

    tr_data, te_data, data_info = generate_nonstationary_data(
        n_points=cli_args.n_points,
        train_ratio=cli_args.train_ratio,
        random_seed=cli_args.seed,
        device=device,
        split_mode=cli_args.split_mode,
    )

    experiments = experiment_specs()
    if cli_args.only_experiment is not None:
        experiments = [exp for exp in experiments if exp[0] == cli_args.only_experiment]
        if not experiments:
            raise ValueError("Unknown experiment name in --only_experiment.")

    results = []
    start_time = time.time()
    for name, att_type, freeze_qk, freeze_v, shared_flow, qk_mode, v_mode, overrides in experiments:
        run_kwargs = vars(cli_args).copy()
        run_kwargs["device"] = str(device)
        run_args = build_args(argparse.Namespace(**run_kwargs))
        for key, value in overrides.items():
            setattr(run_args, key, value)
        results.append(
            run_experiment(
                name,
                att_type,
                run_args,
                tr_data,
                te_data,
                data_info["n_train"],
                data_info["input_dim"],
                data_info["output_dim"],
                device,
                freeze_qk,
                freeze_v,
                shared_flow,
                qk_mode,
                v_mode,
            )
        )

    settings = {
        "seed": cli_args.seed,
        "n_points": cli_args.n_points,
        "n_train": data_info["n_train"],
        "n_test": data_info["n_test"],
        "split_mode": cli_args.split_mode,
        "epochs": cli_args.epochs,
        "frska_extra_epochs": cli_args.frska_extra_epochs,
        "lr": cli_args.lr,
        "eval_batch_size": cli_args.eval_batch_size,
        "device": str(device),
        "elapsed_s": float(time.time() - start_time),
    }
    save_outputs(Path(cli_args.output_dir) / f"seed_{cli_args.seed}", settings, results, data_info)


if __name__ == "__main__":
    main()
