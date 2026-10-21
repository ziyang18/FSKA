# FSKA Dual Experiments

This directory is a standalone two-experiment reproduction package for the paper's `OURS` method, exposed uniformly as `FSKA`.

## Structure

- `models/`: standalone local model stack used by the scripts
- `experiments/mnist/`: MNIST classification with `Transformer` and `FSKA`
- `experiments/toy_nonstationary/`: toy non-stationary regression with `Transformer` and `FSKA`

## Method Naming

- External name: `FSKA`
- Internal attention type: `frska`
- Current main setting for `FSKA`: independent kernels across heads (`shared_flow=False`)

## Default Settings

### MNIST

- `n_samples=1000`
- `epochs=20`
- `lr=0.01`
- `eval_batch_size=1`
- network width/depth/head settings follow the existing focused ablation setup

Run:

```bash
python FSKA_code/experiments/mnist/run_mnist.py
```

### Toy Non-stationary

- `n_points=500`
- `split_mode=random`
- `epochs=60`
- `frska_extra_epochs=40`
- `lr=0.01`
- `eval_batch_size=1`

FSKA toy defaults:

- independent multi-head kernels (`shared_flow=False`)
- `qk_mode=no_qk`
- `v_mode=fixed_orth`
- `M_frska=96`
- `kl_lambda_frska=0.001`

Run:

```bash
python FSKA_code/experiments/toy_nonstationary/run_toy.py
```

## Outputs

Each experiment writes to its own `results/seed_<seed>/` directory and keeps the outputs separated.

### MNIST outputs

- `results_seed.json`
- `ablation_report.txt`
- `training_curves.png`
- `accuracy_bar.png`
- `per_model/*.json`

### Toy outputs

- `results_seed.json`
- `ablation_report.txt`
- `plot_repro_data.npz`
- `prediction_grid_all_models.png`
- `mse_bar.png`
- `plots_single/*.png`
- `per_model/*.json`

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{yang2026fska,
  title     = {Flow-based Spectral Kernel Learning for Nonstationary Attention},
  author    = {Yang, Zi and Li, Ying and Lancho, Alejandro and Larese, Dar{\'i}o C. and Olmos, Pablo M. and Zhang, Michael Minyi},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
