# Home Energy Management under Tiered Peak Power Charges

This repository accompanies our [paper](https://web.stanford.edu/~boyd/papers/hem.html) and provides code and data to replicate all results.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python run.py                    # Run everything
uv run python run.py --train            # Train forecasters only
uv run python run.py --experiments      # Run experiments only
uv run python run.py --figures          # Generate figures only
```

Individual experiments can be run with:
```bash
uv run python run.py --baseline         # Run baseline policies
uv run python run.py --prescient        # Run prescient optimization
uv run python run.py --mpc              # Run MPC policies
uv run python run.py --capacity         # Run capacity sweep
uv run python run.py --sensitivity      # Run sensitivity analysis
```

Results are saved to `results/`, figures to `figures/`.

## Citing
If you want to reference our paper in your research, please consider citing us by using the following BibTeX:
```
@misc{perezpineiro2023home,
      title={Home Energy Management under Tiered Peak Power Charges},
      author={David Pérez-Piñeiro and Sigurd Skogestad and Stephen Boyd},
      year={2023},
      eprint={2307.00000},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```
