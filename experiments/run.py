"""Shared CLI. Each suite runs five model seeds by default and saves runs separately."""
import argparse
import copy
from pathlib import Path

from fska.common import ARCHITECTURES, read_config, resolve_device, save_json, validate_config


def main(suite=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if suite is None:
        parser.add_argument("suite", choices=("sentence", "document", "toy"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--prior", choices=("ngsm", "rbf", "none"))
    parser.add_argument("--trunk-layers", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the five configured seeds")
    parser.add_argument("--datasets", nargs="+", help="Subset of the suite's configured datasets")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration; do not train/download")
    args = parser.parse_args()
    suite = suite or args.suite
    config_path = args.config or Path(__file__).resolve().parents[1] / "configs" / f"{suite}.yaml"
    config = copy.deepcopy(read_config(config_path))
    if config["suite"] != suite:
        parser.error("Config suite does not match the selected experiment")
    if args.architecture:
        config["model"]["architecture"] = args.architecture
    if args.prior:
        config["model"]["prior_type"] = args.prior
    if args.trunk_layers is not None:
        config["model"]["trunk_layers"] = args.trunk_layers
    if args.seeds:
        config["seeds"] = args.seeds
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        if suite not in ("sentence", "document") or args.batch_size < 1:
            parser.error("--batch-size must be positive and is only available for text experiments")
        config["training"]["batch_size"] = args.batch_size
    if args.datasets:
        if not set(args.datasets) <= set(config.get("datasets", [])):
            parser.error("--datasets must be a subset of this suite's datasets")
        config["datasets"] = args.datasets
    validate_config(config)
    if args.dry_run:
        import json
        print(json.dumps(config, indent=2))
        return
    from fska.training import run_text, run_toy
    device = resolve_device(args.device)
    architecture = config["model"]["architecture"]
    jobs = []
    for item in config.get("datasets", ["fska"]):
        for seed in config["seeds"]:
            out = args.output_dir / suite / architecture / item / f"seed_{seed}"
            if out.exists() and any(out.iterdir()):
                raise FileExistsError(f"Refusing to overwrite {out}; choose a new --output-dir")
            jobs.append((item, seed, out))
    for item, seed, out in jobs:
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / "resolved_config.json", config)
        if suite in ("sentence", "document"):
            run_text(config, item, seed, args.data_dir, out, device)
        else:
            run_toy(config, seed, out, device)


if __name__ == "__main__":
    main()

