"""Configuration and repeatability shared by all three suites."""
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ARCHITECTURES = ("per_head_full", "hybrid")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(name):
    # CPU/CUDA are the validated paths; MPS may lack operations used by normflows.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def read_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config):
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("seeds must contain distinct integers")
    model = config["model"]
    if model["architecture"] not in ARCHITECTURES:
        raise ValueError("Unknown architecture")
    if model["architecture"] == "hybrid" and not 0 <= model["trunk_layers"] < model["num_flows"]:
        raise ValueError("Hybrid requires 0 <= trunk_layers < num_flows")
    if model["prior_type"] != "ngsm" and model["architecture"] != "hybrid":
        raise ValueError("RBF/none priors are supported only by the optional Hybrid architecture")
    if config["training"]["epochs"] < 1:
        raise ValueError("epochs must be positive")
    if config["suite"] == "document" and config["data"]["max_seq_len"] is not None:
        raise ValueError("Document suite must use full sequences: max_seq_len: null")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def environment():
    import normflows
    return {"python": platform.python_version(), "torch": torch.__version__,
            "normflows": normflows.__version__, "platform": platform.platform(),
            "cuda": torch.version.cuda}

