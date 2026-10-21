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

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from torchvision import datasets, transforms

    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

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
    args.M_frska = 64
    args.prior_var = 0.1
    args.kl_lambda = 0.1
    args.kl_lambda_frska = 0.005
    args.copula_lambda = 1.0
    args.p_norm = 2.0
    args.att_dropout = 0.1
    args.device = cli_args.device
    args.n_samples = cli_args.n_samples
    args.eval_batch_size = cli_args.eval_batch_size
    args.frska_positive_map = True
    args.dot_qk_mode = "normal"
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
            {"kl_lambda_frska": 0.005, "M_frska": 64, "frska_positive_map": True},
        ),
    ]


def load_mnist_data(n_samples: int, random_seed: int, device: torch.device):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    if HAS_TORCHVISION:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        sample_count = min(n_samples, len(full_dataset))
        downsample_indices = np.random.choice(len(full_dataset), sample_count, replace=False)
        downsample_subset = torch.utils.data.Subset(full_dataset, downsample_indices)
        loader = torch.utils.data.DataLoader(downsample_subset, batch_size=len(downsample_subset), shuffle=False)
        all_data, all_labels = next(iter(loader))
        all_data = all_data.squeeze(1).view(len(downsample_subset), -1)
    else:
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        all_data = torch.FloatTensor((mnist.data / 255.0 - 0.1307) / 0.3081)
        all_labels = torch.LongTensor(mnist.target.astype(int))
        sample_count = min(n_samples, len(all_data))
        sample_idx = np.random.choice(len(all_data), sample_count, replace=False)
        all_data = all_data[sample_idx]
        all_labels = all_labels[sample_idx]

    n_train = int(len(all_data) * 0.7)
    perm = np.random.permutation(len(all_data))
    tr_idx = perm[:n_train]
    te_idx = perm[n_train:]

    train_data = all_data[tr_idx].to(device)
    train_labels = all_labels[tr_idx].to(device)
    test_data = all_data[te_idx].to(device)
    test_labels = all_labels[te_idx].to(device)

    train_mask = torch.ones(train_data.shape[0], 1, dtype=torch.float32, device=device)
    test_mask = torch.ones(test_data.shape[0], 1, dtype=torch.float32, device=device)

    tr_data = (train_data, F.one_hot(train_labels, num_classes=10).float(), train_mask, np.arange(train_data.shape[0]))
    te_data = (test_data, F.one_hot(test_labels, num_classes=10).float(), test_mask, np.arange(test_data.shape[0]))
    return tr_data, te_data, train_data.shape[0], 784, 10, test_labels.detach().cpu().numpy()


class ClassificationTransformer(baseTransformer):
    def __init__(
        self,
        args,
        num_data,
        input_dim,
        output_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        freeze_qk=False,
        freeze_v=False,
    ):
        super().__init__(
            args,
            num_data,
            input_dim,
            output_dim,
            hid_dim,
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            device,
            freeze_qk=freeze_qk,
            freeze_v=freeze_v,
        )


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_classification(model, data, optimizer, kl_weight: float):
    model.train()
    x, y, mask, _ = data
    optimizer.zero_grad()
    output, kld = model(x)
    output = output.squeeze(0)
    target = y.argmax(dim=-1)
    loss = F.cross_entropy(output, target)
    if torch.isnan(kld) or torch.isinf(kld):
        kld = torch.tensor(0.0, device=output.device)
    loss = loss + kl_weight * torch.clamp(kld, min=0.0, max=1000.0) / torch.sum(mask)
    if torch.isnan(loss) or torch.isinf(loss):
        return output, 0.0, 0.0
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    pred = output.argmax(dim=-1)
    acc = (pred == target).float().mean().item()
    return output, loss.item(), acc


def evaluate_classification(model, data, eval_batch_size: int = 1):
    model.eval()
    with torch.no_grad():
        x, y, _, _ = data
        target = y.argmax(dim=-1)
        if eval_batch_size <= 0:
            output, _ = model(x)
            output = output.squeeze(0)
        else:
            outputs = []
            for start in range(0, x.shape[0], eval_batch_size):
                end = min(start + eval_batch_size, x.shape[0])
                batch_output, _ = model(x[start:end])
                outputs.append(batch_output.squeeze(0))
            output = torch.cat(outputs, dim=0)
        pred = output.argmax(dim=-1)
        acc = (pred == target).float().mean().item()
    return output, acc


def run_experiment(exp_name, att_type, args, tr_data, te_data, num_data, input_dim, output_dim, device, freeze_qk, freeze_v, shared_flow, qk_mode, v_mode):
    args.att_type = att_type
    if att_type == "frska":
        args.frska_shared_flow = shared_flow
        args.frska_qk_mode = qk_mode
        args.frska_v_mode = v_mode

    original_m = args.M
    if att_type == "frska":
        args.M = args.M_frska

    model = ClassificationTransformer(
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
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE * 0.5, weight_decay=1e-4)
        kl_weight = getattr(args, "kl_lambda_frska", 0.005)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
        kl_weight = getattr(args, "kl_lambda", 0.1)

    num_params = count_parameters(model)
    best_test_acc = 0.0
    best_epoch = 0
    best_predictions = None
    train_acc_list = []
    test_acc_list = []
    start_time = time.time()

    epoch_iter = range(args.N_EPOCHS)
    if HAS_TQDM:
        epoch_iter = tqdm(epoch_iter, desc=exp_name, ncols=100, leave=False)

    for epoch in epoch_iter:
        _, _, train_acc = train_classification(model, tr_data, optimizer, kl_weight)
        test_pred, test_acc = evaluate_classification(model, te_data, args.eval_batch_size)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_predictions = test_pred.detach().cpu().numpy()
        if HAS_TQDM:
            epoch_iter.set_postfix(best=f"{best_test_acc:.3f}", te=f"{test_acc:.3f}")

    return {
        "name": exp_name,
        "att_type": att_type,
        "best_test_acc": float(best_test_acc),
        "best_epoch": int(best_epoch),
        "runtime": float(time.time() - start_time),
        "num_params": int(num_params),
        "predictions": best_predictions if best_predictions is not None else np.zeros((te_data[0].shape[0], output_dim)),
        "train_acc_list": train_acc_list,
        "test_acc_list": test_acc_list,
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


def plot_training_curves(results, out_path: str):
    plt.figure(figsize=(8, 5))
    for result in results:
        epochs = np.arange(1, len(result["test_acc_list"]) + 1)
        plt.plot(epochs, result["test_acc_list"], linewidth=2.0, label=result["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("MNIST Test Accuracy Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_accuracy_bar(results, out_path: str):
    plt.figure(figsize=(6, 4.5))
    names = [r["name"] for r in results]
    accs = [r["best_test_acc"] for r in results]
    bars = plt.bar(names, accs, color=["#2563eb", "#059669"])
    plt.ylabel("Best Test Accuracy")
    plt.title("MNIST Best Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2.0, acc, f"{acc:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_report(path: str, settings: dict, results: list[dict]):
    lines = [
        "=" * 80,
        "FSKA Dual-Model MNIST Report",
        "=" * 80,
        "",
        "Run setup:",
        f"  seed: {settings['seed']}",
        f"  n_samples: {settings['n_samples']}",
        f"  epochs: {settings['epochs']}",
        f"  learning_rate: {settings['lr']}",
        f"  eval_batch_size: {settings['eval_batch_size']}",
        f"  device: {settings['device']}",
        "",
        "-" * 80,
        f"{'Model':<16} {'Best Acc':<12} {'Best Epoch':<12}",
        "-" * 80,
    ]
    for result in results:
        lines.append(f"{result['name']:<16} {result['best_test_acc']:<12.4f} {result['best_epoch']:<12}")
    lines.extend(
        [
            "",
            "Notes:",
            "- FSKA is the paper's OURS method.",
            "- FSKA here uses independent kernels across heads (shared_flow=False).",
            "- Only Transformer and FSKA are included.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_outputs(out_dir: Path, settings: dict, results: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results_seed.json", "w", encoding="utf-8") as f:
        json.dump({"settings": _to_serializable(settings), "results": _to_serializable(results)}, f, ensure_ascii=False, indent=2)
    write_report(str(out_dir / "ablation_report.txt"), settings, results)
    plot_training_curves(results, str(out_dir / "training_curves.png"))
    plot_accuracy_bar(results, str(out_dir / "accuracy_bar.png"))
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(exist_ok=True)
    for index, result in enumerate(results, start=1):
        with open(per_model_dir / f"{index:02d}_{result['name']}.json", "w", encoding="utf-8") as f:
            json.dump(_to_serializable(result), f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
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

    tr_data, te_data, num_data, input_dim, output_dim, test_labels = load_mnist_data(
        n_samples=cli_args.n_samples,
        random_seed=cli_args.seed,
        device=device,
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
                num_data,
                input_dim,
                output_dim,
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
        "n_samples": cli_args.n_samples,
        "epochs": cli_args.epochs,
        "lr": cli_args.lr,
        "eval_batch_size": cli_args.eval_batch_size,
        "device": str(device),
        "test_size": int(len(test_labels)),
        "elapsed_s": float(time.time() - start_time),
    }
    save_outputs(Path(cli_args.output_dir) / f"seed_{cli_args.seed}", settings, results)


if __name__ == "__main__":
    main()
