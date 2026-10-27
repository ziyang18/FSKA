"""Training routines used by the three public experiment entry points."""
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

from fska.common import environment, save_json, seed_everything
from fska.data.text import NUM_CLASSES, TextDataset, collate_text, load_splits, make_vocab
from fska.data.toy import generate_nonstationary_data
from fska.models.text_model import OursClassifier
from fska.models.vector_model import FSKAVectorModel


def classification_metrics(labels, predictions, num_classes):
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "f1_macro": float(f1_score(labels, predictions, average="macro",
                                       labels=list(range(num_classes)), zero_division=0))}


def require_finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}; stopping this run")


def write_history(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


@torch.no_grad()
def evaluate_text(model, loader, device, n_classes):
    model.eval()
    predictions, targets = [], []
    for ids, mask, labels in loader:
        logits, _ = model(ids.to(device), mask.to(device))
        require_finite(logits, "evaluation logits")
        predictions.extend(logits.argmax(-1).cpu().tolist())
        targets.extend(labels.tolist())
    return classification_metrics(targets, predictions, n_classes), predictions, targets


def run_text(config, dataset, seed, data_dir, out, device):
    seed_everything(seed)
    splits, data_hash = load_splits(data_dir, dataset, config["data"]["split_seed"])
    vocab = make_vocab(splits["train"]["texts"], config["data"]["min_freq"], config["data"]["max_vocab_size"])
    loaders = {}
    for split in ("train", "validation", "test"):
        ds = TextDataset(splits[split], vocab, config["data"]["max_seq_len"])
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=config["training"]["batch_size"], shuffle=split == "train",
            collate_fn=collate_text, num_workers=0,
            generator=torch.Generator().manual_seed(seed))
    model_kwargs = dict(config["model"], vocab_size=len(vocab), num_classes=NUM_CLASSES[dataset],
                        max_seq_len=config["data"]["max_seq_len"] or 256, device=str(device))
    model = OursClassifier(**model_kwargs).to(device)
    t = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["scheduler_t_max"])
    history, best, best_epoch, bad_epochs, best_state = [], -1.0, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for ids, mask, labels in loaders["train"]:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, weighted_kl = model(ids, mask)
            loss = F.cross_entropy(logits, labels) + weighted_kl
            require_finite(loss, "training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
        val, _, _ = evaluate_text(model, loaders["validation"], device, NUM_CLASSES[dataset])
        history.append({"epoch": epoch, "train_loss": loss_sum / count,
                        "validation_accuracy": val["accuracy"], "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"{dataset} seed={seed} epoch={epoch} val_acc={val['accuracy']:.4f}", flush=True)
        if val["accuracy"] > best:
            best, best_epoch, bad_epochs = val["accuracy"], epoch, 0
            if t["checkpoint"] == "best_validation":
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        scheduler.step()
        if t["patience"] is not None and bad_epochs >= t["patience"]:
            break
    if t["checkpoint"] == "best_validation":
        model.load_state_dict(best_state)
    metrics, predictions, labels = evaluate_text(model, loaders["test"], device, NUM_CLASSES[dataset])
    result = {"dataset": dataset, "seed": seed, "architecture": config["model"]["architecture"],
              "metrics": metrics, "epochs_completed": len(history),
              "selected_epoch": best_epoch if t["checkpoint"] == "best_validation" else len(history),
              "checkpoint_rule": t["checkpoint"], "runtime_seconds": time.perf_counter() - started,
              "data_sha256": data_hash, "split_sizes": {k: len(v["texts"]) for k, v in splits.items()},
              "max_observed_tokens": {k: max(map(len, loaders[k].dataset.ids)) for k in loaders},
              "config": config, "environment": environment()}
    torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs, "vocabulary": vocab,
                "config": config, "seed": seed, "data_sha256": data_hash}, out / "model.pt")
    save_json(out / "result.json", result)
    save_json(out / "predictions.json", {"labels": labels, "predictions": predictions})
    write_history(out / "history.csv", history)
    return result


def vector_args(config):
    m = config["model"]
    return SimpleNamespace(
        KEY_DIM=m["hidden_dim"], M=m["M"], kl_lambda_frska=m["kl_lambda"],
        frska_architecture=m["architecture"], frska_num_flows=m["num_flows"],
        frska_num_mixtures=m["num_mixtures"], frska_mlp_hidden_dim=m["mlp_hidden_dim"],
        frska_prior_type=m["prior_type"], frska_trunk_layers=m["trunk_layers"],
        frska_qk_mode="no_qk", frska_v_mode="fixed_orth",
        frska_positive_map=True, frska_return_dense_attention=False)


def make_vector_model(config, device, num_data, input_dim, output_dim):
    m = config["model"]
    return FSKAVectorModel(vector_args(config), num_data, input_dim, output_dim,
                           m["hidden_dim"], m["n_layers"], m["n_heads"], m["hidden_dim"],
                           m["dropout"], device).to(device)


@torch.no_grad()
def predict_vector(model, x, batch_size):
    model.eval()
    chunks = [x] if batch_size <= 0 else x.split(batch_size)
    predictions = [model(chunk)[0].squeeze(0) for chunk in chunks]
    return torch.cat(predictions)


def run_toy(config, seed, out, device):
    suite, t = config["suite"], config["training"]
    if suite != "toy":
        raise ValueError("run_toy requires a toy configuration")
    d = config["data"]
    tr_raw, te_raw, data_info = generate_nonstationary_data(
        d["n_points"], d["train_ratio"], d["split_seed"], "cpu", d["split_mode"])
    tr, te = tr_raw[:2], te_raw[:2]
    input_dim, output_dim = data_info["input_dim"], 1
    # Fixed data/split across all five model seeds; no model constructor resets the RNG.
    seed_everything(seed)
    x, y = (v.to(device) for v in tr)
    tx, ty = (v.to(device) for v in te)
    model = make_vector_model(config, device, len(x), input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    history, started = [], time.perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output, weighted_kl = model(x)
        output = output.squeeze(0)
        task_loss = F.mse_loss(output, y)
        # Explicitly retain the vector runner's original KL convention.
        # Attention already applies lambda; this legacy path weights again and divides by N.
        if t["kl_convention"] == "legacy_double_weight_per_sample":
            penalty = config["model"]["kl_lambda"] * weighted_kl.clamp(0, 1000) / len(x)
        elif t["kl_convention"] == "single_weight":
            penalty = weighted_kl
        else:
            raise ValueError("Unknown KL convention")
        loss = task_loss + penalty
        require_finite(loss, "training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        history.append({"epoch": epoch, "train_task_loss": task_loss.item(), "train_loss": loss.item()})
    prediction = predict_vector(model, tx, t["eval_batch_size"])
    require_finite(prediction, "predictions")
    pred = prediction.squeeze(-1).cpu().numpy()
    target = ty.squeeze(-1).cpu().numpy()
    metrics = {"mse": float(mean_squared_error(target, pred)),
               "mae": float(mean_absolute_error(target, pred)), "r2": float(r2_score(target, pred))}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data_info["time_axis"], data_info["target_clean_full"], label="Clean target")
    ax.scatter(data_info["time_axis"][data_info["test_idx"]], pred, s=10, label="FSKA prediction")
    ax.set(xlabel="t", ylabel="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prediction.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(out / "predictions.npz", target=ty.cpu().numpy(), prediction=pred)
    # Save inputs and split indices so vector-input evaluation is reproducible.
    torch.save({"state_dict": model.state_dict(), "config": config, "variant": "fska",
                "seed": seed, "train_data": tuple(v.cpu() for v in tr),
                "test_data": tuple(v.cpu() for v in te), "data_info": data_info}, out / "model.pt")
    result = {"dataset": suite, "variant": "fska", "seed": seed, "metrics": metrics,
              "architecture": config["model"]["architecture"], "selected_epoch": t["epochs"],
              "checkpoint_rule": "final", "runtime_seconds": time.perf_counter() - started,
              "config": config, "environment": environment()}
    save_json(out / "result.json", result)
    write_history(out / "history.csv", history)
    print(f"{suite}/fska seed={seed}: {metrics}", flush=True)
    return result
