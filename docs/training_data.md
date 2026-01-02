# Training Data Location and W&B Logging

## Where Raw Training Data is Captured Locally

All training run data is stored in the `runs/` directory at the project root. Each training run creates a timestamped directory:

```
runs/deformable_detr_table_detector_minimal YYYY-MM-DD HH:MM/
```

### Directory Structure

Each run directory contains:

1. **`checkpoints/`** - Model checkpoints (saved periodically and for best model)
   - `checkpoint_epoch_{N}.pt` - Periodic checkpoints (every `save_every_n_epochs` epochs)
   - `best.pt` - Best model checkpoint (based on validation F1 score)
   - Each checkpoint contains:
     - `model_state_dict` - Model weights
     - `optimizer_state_dict` - Optimizer state
     - `scheduler_state_dict` - Learning rate scheduler state
     - `epoch` - Epoch number
     - `train_metrics` - Training metrics at that epoch
     - `val_metrics` - Validation metrics at that epoch
     - `best_val_f1` - Best validation F1 score so far

2. **`configs/`** - Configuration files
   - `config.yaml` - Complete training configuration (experiment, data, model, training, evaluation parameters)

3. **`model_arch.txt`** - Textual representation of the model architecture

4. **`wandb/`** - Local W&B run data (synced to cloud)
   - `run-{timestamp}-{run_id}/` - W&B run directory
     - `run-{run_id}.wandb` - Binary W&B log file
     - `files/wandb-summary.json` - Run summary metrics
     - `files/wandb-metadata.json` - Run metadata

### Example Path

```
runs/deformable_detr_table_detector_minimal 2025-12-10 13:13/
├── checkpoints/
│   └── (checkpoint files if saved)
├── configs/
│   └── config.yaml
├── model_arch.txt
└── wandb/
    └── run-20251210_131357-{run_id}/
        ├── run-{run_id}.wandb
        └── files/
            ├── wandb-summary.json
            └── wandb-metadata.json
```

## What Data is Sent to W&B (Weights & Biases)

### 1. Configuration/Hyperparameters (`wandb.config`)

All configuration parameters are logged to W&B config:

- **Experiment settings**: name, project_name
- **Data settings**: 
  - `data/train_size` - Number of training samples
  - `data/val_size` - Number of validation samples
  - `data/datasets` - List of dataset names used
- **Model settings**:
  - `model/total_params` - Total model parameters
  - `model/trainable_params` - Trainable parameters
  - `model/total_params_m` - Parameters in millions
  - All model architecture parameters (backbone, DETR config, loss weights)
- **Training settings**: batch_size, lr, weight_decay, max_epochs, etc.

### 2. Metrics Logged During Training (`wandb.log`)

#### Training Metrics (per epoch):
- `train/loss` - Total training loss
- `train/det_loss` - Detection loss (classification + regression)
- `train/cls_loss` - Classification loss component
- `train/l1_loss` - L1 bounding box regression loss
- `train/iou_loss` - IoU/GIoU loss component
- `train/lr` - Current learning rate
- `train/loss_ratio_cls_l1` - Ratio of classification to L1 loss (for debugging)

#### Validation Metrics (per epoch):
- `val/loss` - Total validation loss
- `val/precision` - Precision score
- `val/recall` - Recall score
- `val/f1` - F1 score
- `val/eob_mean` - Mean End-of-Block (EoB) metric (optional)

#### System Metrics (per epoch):
- `system/gpu_memory_mb` - GPU memory usage (if available)
- `system/cpu_memory_mb` - CPU memory usage
- `system/epoch_time_seconds` - Time taken per epoch

#### Gradient Metrics (per gradient accumulation step):
- `train/max_grad_norm` - Maximum gradient norm across all parameters
- `train/mean_grad_norm` - Mean gradient norm across all parameters

### 3. Model Architecture Artifact

- **Artifact Name**: `{experiment_name}_architecture` (sanitized)
- **Type**: `model_architecture`
- **Content**: `model_arch.txt` file containing the textual representation of the model

### 4. Model Checkpoints (if logged as artifacts)

- Model checkpoints can be logged as W&B artifacts (currently not automatically logged, but can be added)
- Would include: `best.pt` and periodic checkpoints

### 5. Gradient Histograms

- W&B watches the model and logs gradient histograms automatically (`wandb.watch(model, log="gradients", log_freq=100)`)
- This tracks gradient distributions for debugging training stability

### 6. Run Summary (`wandb.run.summary`)

Final summary metrics logged at the end:
- `best_epoch` - Epoch with best validation F1
- `best_{metric}` - Best values for each metric (e.g., `best_f1`, `best_precision`)

## What is NOT Sent to W&B

The following data stays **local only**:

1. **Raw training data** (Excel files) - Never uploaded
2. **Feature tensors** - Only metrics are logged, not the actual feature data
3. **Ground truth annotations** - Only statistics (counts) are logged
4. **Intermediate model outputs** - Only final metrics are logged
5. **Full training logs/console output** - Only metrics are logged

## Accessing W&B Data

1. **Web Dashboard**: Visit https://wandb.ai and navigate to your project (`tablesense2`)
2. **API**: Use `wandb.Api()` to programmatically access runs and metrics
3. **Local Files**: W&B data is also cached locally in the `wandb/` subdirectory of each run

## Notes

- W&B runs are synced to the cloud automatically (unless `mode="offline"`)
- The local `wandb/` directory contains cached data that can be synced later if needed
- Model checkpoints are saved locally but not automatically uploaded as artifacts (can be added if needed)
- All metrics are logged per epoch, making it easy to track training progress over time
