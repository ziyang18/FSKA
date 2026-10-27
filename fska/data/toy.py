import numpy as np
import torch

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
