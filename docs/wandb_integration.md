# W&B Sweep Integration Plan

This document outlines the refined solution for integrating Weights & Biases sweeps into the table detection workflow. This approach ensures robust configuration management, reproducibility, and resource efficiency.

## 1. W&B Sweep Configuration

Create `configs/sweep_config.yaml`.

**Key Features:**
*   **Explicit Command:** Ensures the base config (`detector_config.yaml`) is loaded first.
*   **Fixed Evaluation Metric:** Removed metric thresholds from the sweep to ensure F1 scores are comparable across runs.
*   **Warmup-Aware Pruning:** `min_iter` is set higher than `warmup_epochs` to avoid killing runs before they stabilize.

```yaml
# configs/sweep_config.yaml
# W&B Sweep Configuration
name: "detr_hyperparam_tuning_v1"
program: scripts/train_detector.py
method: bayes  # Bayesian optimization for efficiency
metric:
  name: val/f1
  goal: maximize

# Explicit command to load the base config and then override with sweep params
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--config"
  - "configs/detector_config.yaml"
  - "--project-root"
  - "."
  - "--sweep"
  - ${args}

parameters:
  # Training Dynamics
  training.lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  
  training.weight_decay:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  
  training.warmup_epochs:
    values: [3, 5]
    
  training.gradient_accumulation_steps:
    values: [4, 8, 16]

  # DETR Architecture (Constrained to avoid OOM)
  model.detr.hidden_dim:
    values: [128, 256]
  
  model.detr.num_encoder_layers:
    values: [4, 6]
    
  model.detr.num_decoder_layers:
    values: [4, 6]
    
  model.detr.dropout:
    values: [0.1, 0.2, 0.3]

  # Loss Weights (Crucial for DETR convergence)
  model.loss.lambda_cls:
    min: 0.5
    max: 2.0
  
  model.loss.lambda_l1:
    min: 2.0
    max: 10.0
  
  model.loss.lambda_iou:
    min: 1.0
    max: 5.0

# Early termination to prune bad runs, respecting warmup
early_terminate:
  type: hyperband
  min_iter: 6  # > max(warmup_epochs) to allow convergence start
  eta: 2
```

## 2. Enhanced Experiment Logger

Update `training/experiment_logger.py` to handle sweep parameters and model summaries without external dependencies.

```python
# training/experiment_logger.py
# ... existing imports ...
import wandb
import torch.nn as nn

class ExperimentLogger:
    # ... keep existing __init__ ...

    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters explicitly for sweeps.
        """
        wandb.config.update(hyperparams, allow_val_change=True)

    def log_model_summary(self, model: nn.Module):
        """
        Log model architecture summary and parameter counts.
        """
        # Robust parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.config.update({
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
            'model/total_params_m': total_params / 1e6,
        })
        
        # Log textual representation as a persistent artifact
        # This is safer than torchsummary for complex/custom architectures
        with open("model_arch.txt", "w") as f:
            f.write(str(model))
        
        artifact = wandb.Artifact(
            name=f"{self.experiment_name}_architecture",
            type="model_architecture"
        )
        artifact.add_file("model_arch.txt")
        self.wandb_run.log_artifact(artifact)
        
        # Optional: Watch gradients
        wandb.watch(model, log="gradients", log_freq=100)

    def log_dataset_stats(self, train_size: int, val_size: int, dataset_names: list):
        """
        Log dataset statistics context.
        """
        wandb.config.update({
            'data/train_size': train_size,
            'data/val_size': val_size,
            'data/datasets': dataset_names,
        })
```

## 3. Sweep-Aware Training Script

Update `scripts/train_detector.py` to merge the sweep configuration safely into your base config.

```python
# scripts/train_detector.py
# ... imports ...
import copy

def _merge_sweep_config(base_config: dict, sweep_config: dict) -> dict:
    """
    Merge W&B sweep config into base config with nested key support.
    Example: sweep key 'training.lr' updates base_config['training']['lr']
    """
    merged = copy.deepcopy(base_config)
    for key, value in sweep_config.items():
        # Handle nested keys (e.g., "training.lr")
        if '.' in key:
            parts = key.split('.')
            target = merged
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            merged[key] = value
    return merged

def main():
    parser = argparse.ArgumentParser(description="Train Deformable DETR table detector")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--sweep", action="store_true", help="Run as part of W&B sweep")
    
    args = parser.parse_args()
    
    # 1. Load Base Config
    cfg = Config.from_yaml(args.config)
    config = cfg.to_dict()
    
    # 2. Merge Sweep Config if active
    if args.sweep:
        # wandb.init() will pick up the sweep params automatically
        # We initialize early here to get the config
        run = wandb.init(project=config['experiment'].get('project_name', 'tablesense2'))
        config = _merge_sweep_config(config, dict(wandb.config))
        
        # Update run name to be descriptive
        run.name = f"sweep_{run.id}"
        
        # We must pass the `run` object or config to the Logger to avoid re-init
    
    # ... Setup paths ...
    
    # ... Create Data Loaders ...
    
    # ... Create Model ...
    
    # 3. Initialize Logger (modified to handle existing run)
    if args.sweep:
        # If sweep, we already initialized wandb above.
        # pass existing config and rely on global wandb state
        logger = ExperimentLogger(
            experiment_name=config['experiment']['name'],
            config=config,
            resume=wandb.run.id 
        )
    else:
        # Normal run
        logger = ExperimentLogger(
            experiment_name=config['experiment']['name'],
            config=config
        )

    # 4. Enhanced Logging
    logger.log_model_summary(model)
    logger.log_dataset_stats(
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        dataset_names=config['data']['dataset_names']
    )

    # ... Rest of training loop ...
```

## 4. Simplified Launch Script

Create `scripts/launch_sweep.py` to easily start the controller.

```python
# scripts/launch_sweep.py
"""
Launch W&B sweep for hyperparameter tuning.
"""
import wandb
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch W&B sweep")
    parser.add_argument("--config", type=str, default="configs/sweep_config.yaml")
    parser.add_argument("--count", type=int, default=20, help="Max number of runs")
    parser.add_argument("--project", type=str, default="tablesense2")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize sweep controller
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    
    print(f"Sweep ID: {sweep_id}")
    print(f"Start agent with: wandb agent {args.project}/{sweep_id}")
    
    # Optional: Start agent immediately in this process
    # wandb.agent(sweep_id, function=train_function, count=args.count)

if __name__ == "__main__":
    main()
```

## 5. Trainer Updates

In `training/trainer.py`, add simple logic to log key training dynamics.

```python
# training/trainer.py
# Inside train() loop:
    # Log Learning Rate (simple, effective)
    current_lr = self.optimizer.param_groups[0]['lr']
    
    # Calculate ratios for debugging DETR stability
    logs = {'train/lr': current_lr}
    
    # Log detailed loss breakdown if available
    if 'cls_loss' in train_metrics and 'l1_loss' in train_metrics:
        # Monitor if classification overpowers regression or vice versa
        logs['train/loss_ratio_cls_l1'] = train_metrics['cls_loss'] / (train_metrics['l1_loss'] + 1e-6)

    self.logger.log_metrics(logs, step=epoch)
```

