# Experiments and Tracking

TableSense2 integrates with **Weights & Biases (W&B)** for experiment tracking, logging metrics, hyperparameters, and artifacts.

---

## Weights & Biases Setup

1. Create an account at https://wandb.ai
2. Login:

```bash
wandb login
```

---

## Logged Information

The experiment logger automatically tracks:

- Training losses (total and per-component)
- Validation metrics (Precision, Recall, F1, EoB)
- Learning rate schedules
- Model checkpoints
- Dataset statistics
- System metrics (optional)

---

## Modes

W&B behavior can be controlled via the logger mode:

- `online` (default)
- `offline`
- `disabled`

```python
from training.experiment_logger import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="my_experiment",
    mode="offline"
)
```

---

## Hyperparameter Sweeps

The project includes native support for **W&B Sweeps** using Bayesian optimization.

### Launching a Sweep

```bash
python scripts/launch_sweep.py --config configs/sweep_config.yaml
```

### Running Sweep Agents

```bash
wandb agent <project>/<sweep_id>
```

The sweep configuration defines:
- optimizer parameters
- architecture dimensions
- DETR loss weights

Hyperband early stopping is enabled to conserve compute.

---

## Best Practices

- Start with small sweeps (10–20 runs)
- Validate configs locally before scaling
- Review parameter importance in the W&B UI

