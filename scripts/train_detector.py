"""
Training script for Deformable DETR table detector.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import copy
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import wandb


from models.table_detector import TableDetector
from training.trainer import Trainer
from training.experiment_logger import ExperimentLogger
from training.config import Config
from training.visualization_selector import VisualizationSelector
from data_io.annotations import AnnotationLoader
from features.featurizer import CellFeaturizer
from evaluation.hungarian_evaluator import HungarianEvaluator
from utils.excel_utils import load_workbook

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _get_default_device() -> str:
    """
    Select a reasonable default device:
    - Prefer CUDA if available
    - Otherwise prefer Apple MPS (for Apple Silicon Macs)
    - Fall back to CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    # MPS is the Metal Performance Shaders backend used on Apple Silicon (M1/M2/…)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"



class DETRDataset(Dataset):
    """
    Dataset for DETR table detector training.
    
    Loads Excel sheets, featurizes them, and formats targets for DETR loss computation.
    """
    
    def __init__(
        self,
        annotations: list,
        project_root: Path,
        featurizer: CellFeaturizer,
    ):
        """
        Initialize dataset.
        
        Args:
            annotations: List of annotation dictionaries
            project_root: Root directory of the project
            featurizer: CellFeaturizer instance
        """
        self.project_root = project_root
        self.featurizer = featurizer
        
        # Filter out annotations with missing files and validate they can be loaded
        self.annotations = []
        skipped_missing = 0
        skipped_corrupted = 0
        
        for ann in annotations:
            if not self._file_exists(ann):
                skipped_missing += 1
                continue
            
            # Try to validate the file can be loaded (quick check)
            if not self._validate_file(ann):
                skipped_corrupted += 1
                continue
            
            self.annotations.append(ann)
        
        if skipped_missing > 0:
            print(f"Warning: Skipped {skipped_missing} annotations with missing files")
        if skipped_corrupted > 0:
            print(f"Warning: Skipped {skipped_corrupted} annotations with corrupted/unreadable files")
    
    def _file_exists(self, ann: dict) -> bool:
        """
        Check if the file for an annotation exists (with flexible matching).
        
        Returns:
            True if file exists, False otherwise
        """
        base_path = self.project_root / ann['file_path']
        file_name = ann['file_name']
        file_path = base_path / file_name
        
        # Check exact match first
        if file_path.exists():
            return True
        
        # Try to find file ending with the annotation filename
        # List all files and check if any ends with the annotation filename
        try:
            all_files = list(base_path.glob("*.xlsx"))
            for f in all_files:
                if f.name.endswith(file_name):
                    return True
        except Exception:
            pass
        
        return False
    
    def _validate_file(self, ann: dict) -> bool:
        """
        Validate that the file can be loaded and the sheet exists.
        Uses the same loading method as __getitem__ to catch the same errors.
        
        Returns:
            True if file is valid and readable, False otherwise
        """
        try:
            base_path = self.project_root / ann['file_path']
            file_name = ann['file_name']
            file_path = base_path / file_name
            
            # Try to find matching file if exact match doesn't exist
            if not file_path.exists():
                all_files = list(base_path.glob("*.xlsx"))
                matching_file = None
                for f in all_files:
                    if f.name.endswith(file_name):
                        matching_file = f
                        break
                if matching_file:
                    file_path = matching_file
                else:
                    return False
            
            # Try to load the workbook using the same method as __getitem__
            # Use data_only=False (same as training) to catch the same errors
            # Don't use read_only=True as it might not catch all XML parsing errors
            workbook = load_workbook(file_path, data_only=False)
            
            # Check if sheet exists
            sheet_name = ann['sheet_name']
            if sheet_name not in workbook.sheetnames:
                workbook.close()
                return False
            
            # Try to access the sheet to ensure it's readable
            # This will catch XML parsing errors that occur when accessing sheet data
            # Some errors only occur when reading the sheet content, not just the workbook
            sheet = workbook[sheet_name]
            # Try to access sheet properties to trigger any XML parsing errors
            try:
                _ = sheet.max_row
                _ = sheet.max_column
                # Try to iterate over a few cells to catch parsing errors
                # This mimics what featurizer will do
                for row in sheet.iter_rows(min_row=1, max_row=min(10, sheet.max_row), values_only=False):
                    if row:
                        break
            except Exception:
                workbook.close()
                return False
            
            workbook.close()
            return True
        except Exception:
            # File is corrupted or unreadable
            return False
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load workbook
        # Handle case where annotation filename doesn't match actual filename exactly
        # (e.g., annotation has "1_file.xlsx" but actual file is "VEnron2_VEnron2_253_1_file.xlsx")
        base_path = self.project_root / ann['file_path']
        file_name = ann['file_name']
        file_path = base_path / file_name
        
        # If exact match doesn't exist, try to find file ending with the annotation filename
        if not file_path.exists():
            # List all files and find one that ends with the annotation filename
            all_files = list(base_path.glob("*.xlsx"))
            matching_file = None
            for f in all_files:
                if f.name.endswith(file_name):
                    matching_file = f
                    break
            
            if matching_file:
                file_path = matching_file
            else:
                # This should not happen if _file_exists worked correctly, but handle it anyway
                raise FileNotFoundError(
                    f"Excel file not found: {file_path}\n"
                    f"Annotation file_name: {file_name}\n"
                    f"Searched in: {base_path}"
                )
        
        try:
            workbook = load_workbook(file_path, data_only=False)
        except Exception as e:
            # Log the corrupted file and re-raise with better context
            import warnings
            warnings.warn(
                f"Skipping corrupted file: {file_path} (sheet: {ann['sheet_name']})\n"
                f"Error: {str(e)}\n"
                f"This file should have been filtered during validation. "
                f"Consider removing it from annotations or fixing the file.",
                UserWarning
            )
            raise RuntimeError(
                f"Failed to load workbook: {file_path}\n"
                f"File name: {ann['file_name']}\n"
                f"Sheet name: {ann['sheet_name']}\n"
                f"Error: {str(e)}\n"
                f"Note: This file passed initial validation but failed during training. "
                f"The file may be corrupted or have XML parsing issues."
            ) from e
        
        sheet_name = ann['sheet_name']
        
        # Featurize sheet
        try:
            features, metadata = self.featurizer.featurize_sheet(workbook, sheet_name)
        except Exception as e:
            workbook.close()
            import warnings
            warnings.warn(
                f"Failed to featurize sheet: {file_path} / {sheet_name}\n"
                f"Error: {str(e)}",
                UserWarning
            )
            raise RuntimeError(
                f"Failed to featurize sheet: {file_path} / {sheet_name}\n"
                f"Error: {str(e)}"
            ) from e
        H, W = features.shape[:2]
        
        # Get sheet dimensions for normalization
        sheet_height = metadata.get('height', H)
        sheet_width = metadata.get('width', W)
        
        # Convert GT boxes to normalized (cx, cy, w, h) format
        # GT boxes are in 1-based (col_left, row_top, col_right, row_bottom)
        gt_boxes_normalized = []
        gt_labels = []
        
        for col_left, row_top, col_right, row_bottom in ann['table_regions_numeric']:
            # Convert to 0-based for computation
            x1 = (col_left - 1) / sheet_width
            y1 = (row_top - 1) / sheet_height
            x2 = col_right / sheet_width
            y2 = row_bottom / sheet_height
            
            # Clamp to [0, 1]
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            
            # Convert to (cx, cy, w, h)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            
            # Skip degenerate boxes
            if w > 0 and h > 0:
                gt_boxes_normalized.append([cx, cy, w, h])
                gt_labels.append(1)  # 1 = table class, 0 = background
        
        # Create padding mask (all valid for variable-sized sheets, pass None to let backbone handle it)
        mask = None  # All positions are valid, backbone will create appropriate masks if needed
        
        # Convert features to tensor: (H, W, C) -> (C, H, W)
        features_tensor = torch.from_numpy(features).permute(2, 0, 1).float()
        
        # Create target dict for DETR
        target = {
            'labels': torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.zeros((0,), dtype=torch.long),
            'boxes': torch.tensor(gt_boxes_normalized, dtype=torch.float32) if gt_boxes_normalized else torch.zeros((0, 4), dtype=torch.float32),
        }
        
        workbook.close()
        
        return {
            'features': features_tensor,
            'mask': mask,
            'targets': target,
            'gt_boxes': ann['table_regions_numeric'],  # Keep original for evaluation
            'workbook_id': ann['workbook_id'],
            'sheet_name': sheet_name,
            'file_path': ann['file_path'],
            'file_name': ann['file_name'],
            'sheet_height': sheet_height,
            'sheet_width': sheet_width,
        }


class DETRTrainer(Trainer):
    """
    Extended trainer that handles DETR-style loss dictionaries.
    """
    
    def __init__(self, *args, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize DETR trainer with optional config for saving.
        """
        super().__init__(*args, **kwargs)
        self.config = config
        
        # Initialize visualization selector if config provided
        self.viz_selector = None
        self.viz_start_epoch = 25
        if config and 'visualization' in config:
            viz_config = config['visualization']
            self.viz_start_epoch = viz_config.get('start_epoch', 25)
            bucket_config = viz_config.get('buckets', {})
            cache_epochs = viz_config.get('cache_epochs', 3)
            self.viz_selector = VisualizationSelector(
                best_count=bucket_config.get('best_count', 20),
                worst_count=bucket_config.get('worst_count', 20),
                high_miss_count=bucket_config.get('high_miss_count', 20),
                high_extra_count=bucket_config.get('high_extra_count', 20),
                cache_epochs=cache_epochs,
            )
    
    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Compute loss for DETR model.
        
        DETR models return a dictionary with 'losses' key containing loss components.
        """
        if isinstance(batch, dict):
            features = batch['features'].to(self.device)
            # Add batch dimension if missing (batch_size=1 case)
            if features.dim() == 3:
                features = features.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            
            mask = batch.get('mask')
            targets = batch.get('targets')
            
            # Mask is None for variable-sized sheets (all positions valid)
            # If mask is provided, ensure it has batch dimension
            if mask is not None:
                mask = mask.to(self.device)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
            
            # Move targets to device
            if targets is not None:
                # Wrap single target in list (batch_size=1)
                targets_device = [{
                    'labels': targets['labels'].to(self.device),
                    'boxes': targets['boxes'].to(self.device),
                }]
            
                # Forward pass with targets (training mode)
                model_output = self.model(x=features, mask=mask, targets=targets_device)
            else:
                # Inference mode
                model_output = self.model(x=features, mask=mask, targets=None)
            
            # Handle loss dictionary (DETR-style)
            if isinstance(model_output, dict):
                if 'losses' in model_output:
                    # Extract total loss from losses dict
                    losses = model_output['losses']
                    if isinstance(losses, dict):
                        # Sum all loss components
                        total_loss = sum(losses.values())
                        # Store loss components for logging
                        self._last_loss_components = losses
                        return total_loss
                    else:
                        return losses
                elif 'loss' in model_output:
                    return model_output['loss']
                else:
                    raise ValueError("Model returned dict but missing 'loss' or 'losses' key")
            else:
                raise ValueError("DETR model should return a dictionary with losses")
        else:
            raise ValueError("Batch format not supported. Expected dict.")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch, logging DETR loss components.
        """
        metrics = super().train_epoch()
        
        # Add DETR loss components if available
        if hasattr(self, '_last_loss_components') and self._last_loss_components:
            loss_components = self._last_loss_components
            # Extract common loss names
            if 'loss_ce' in loss_components:
                metrics['cls_loss'] = loss_components['loss_ce'].item()
            if 'loss_bbox' in loss_components:
                metrics['l1_loss'] = loss_components['loss_bbox'].item()
            if 'loss_giou' in loss_components:
                metrics['iou_loss'] = loss_components['loss_giou'].item()
            
            # Note: We don't log here anymore - train() method handles all logging
            # to avoid duplicate logging at the same step
        
        return metrics
    
    def train(self, evaluator: Optional[Any] = None, early_stopping_patience: Optional[int] = None):
        """
        Main training loop with bucket-based visualization logging.
        
        Overrides base train to enable bucket-based visualization starting at epoch 25.
        """
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Determine if we should collect visualization data (bucket-based, starting at epoch 25)
            log_viz = (epoch >= self.viz_start_epoch) and (self.viz_selector is not None)
            
            # Validate (with per-image statistics collection if requested)
            val_metrics = self.validate(evaluator=evaluator, log_visualizations=log_viz)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Bucket-based visualization: select and log images
            if log_viz and 'per_image_data' in val_metrics and self.viz_selector is not None and self.logger:
                per_image_data = val_metrics['per_image_data']
                if per_image_data:
                    # Select images from buckets
                    selected_images = self.viz_selector.select_images(per_image_data, epoch)
                    
                    # Log selected images to W&B
                    step_for_epoch = len(self.train_loader) * epoch + (len(self.train_loader) - 1)
                    for bucket, images in selected_images.items():
                        for img_data in images:
                            stats = img_data['stats']
                            batch_data = img_data['batch_data']
                            
                            log_prediction_image(
                                sheet_features=batch_data['features'],
                                gt_boxes=batch_data['gt_boxes'],
                                pred_boxes=batch_data['pred_boxes'],
                                epoch=epoch,
                                sheet_name=img_data['sheet_name'],
                                file_name=img_data['file_name'],
                                matched_pairs=stats.get('matched_pairs', []),
                                unmatched_gt_indices=stats.get('unmatched_gt_indices', []),
                                unmatched_pred_indices=stats.get('unmatched_pred_indices', []),
                                metrics_dict=stats,
                                bucket=bucket,
                                global_step=step_for_epoch,
                            )
                    
                    # Update cache
                    self.viz_selector.update_cache(epoch)
            
            # Log metrics - combine all metrics into a single wandb.log() call per epoch
            # to avoid issues with multiple wandb.log() calls at the same step
            if self.logger:
                
                # Build combined metrics dictionary
                combined_metrics = {}
                
                # Loss metrics
                if train_metrics.get('loss') is not None:
                    combined_metrics['train/loss'] = train_metrics['loss']
                if val_metrics.get('loss') is not None:
                    combined_metrics['val/loss'] = val_metrics['loss']
                if train_metrics.get('cls_loss') is not None:
                    combined_metrics['train/cls_loss'] = train_metrics['cls_loss']
                if train_metrics.get('l1_loss') is not None:
                    combined_metrics['train/l1_loss'] = train_metrics['l1_loss']
                if train_metrics.get('iou_loss') is not None:
                    combined_metrics['train/iou_loss'] = train_metrics['iou_loss']
                if train_metrics.get('loss') is not None:
                    combined_metrics['train/det_loss'] = train_metrics['loss']
                
                # Learning rate and loss ratios
                if 'lr' in train_metrics:
                    combined_metrics['train/lr'] = train_metrics['lr']
                if 'cls_loss' in train_metrics and 'l1_loss' in train_metrics:
                    combined_metrics['train/loss_ratio_cls_l1'] = train_metrics['cls_loss'] / (train_metrics['l1_loss'] + 1e-6)
                
                # System metrics
                try:
                    gpu_memory_mb = None
                    if torch.cuda.is_available():
                        gpu_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                        torch.cuda.reset_peak_memory_stats(self.device)
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        gpu_memory_mb = None
                    
                    cpu_memory_mb = None
                    try:
                        import psutil
                        cpu_memory_mb = psutil.Process().memory_info().rss / (1024 ** 2)
                    except ImportError:
                        pass
                    
                    if gpu_memory_mb is not None:
                        combined_metrics['system/gpu_memory_mb'] = gpu_memory_mb
                    if cpu_memory_mb is not None:
                        combined_metrics['system/cpu_memory_mb'] = cpu_memory_mb
                    combined_metrics['system/epoch_time_seconds'] = epoch_time
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to collect system metrics: {e}")
                
                # Validation metrics
                if 'f1' in val_metrics:
                    combined_metrics['val/precision'] = val_metrics.get('precision', 0.0)
                    combined_metrics['val/recall'] = val_metrics.get('recall', 0.0)
                    combined_metrics['val/f1'] = val_metrics.get('f1', 0.0)
                    if val_metrics.get('eob_mean') is not None:
                        combined_metrics['val/eob_mean'] = val_metrics['eob_mean']
                    if val_metrics.get('eob_std') is not None:
                        combined_metrics['val/eob_std'] = val_metrics['eob_std']
                    if val_metrics.get('tp') is not None:
                        combined_metrics['val/tp'] = val_metrics['tp']
                    if val_metrics.get('fp') is not None:
                        combined_metrics['val/fp'] = val_metrics['fp']
                    if val_metrics.get('fn') is not None:
                        combined_metrics['val/fn'] = val_metrics['fn']
                
                # Log all metrics in a single call
                if combined_metrics:
                    # Use the same step calculation as gradient metrics to ensure W&B stores all metrics
                    # Gradient metrics use: len(train_loader) * epoch + batch_idx
                    # Log loss metrics at the last batch step of the epoch to align with gradient logging pattern
                    step_for_epoch = len(self.train_loader) * epoch + (len(self.train_loader) - 1)
                    self.logger.log_metrics(combined_metrics, step=step_for_epoch)
            
            # Save checkpoint
            val_f1 = val_metrics.get('f1', 0.0)
            is_best = val_f1 > best_val_f1
            
            if is_best:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.checkpoint_dir:
                # Save periodic checkpoint
                if (epoch + 1) % self.save_every_n_epochs == 0:
                    self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=False)
                
                # Save best checkpoint
                if is_best:
                    self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
            
            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        print(f"Training completed. Best validation F1: {best_val_f1:.4f}")
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint with config."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_f1': self.best_val_f1,
        }
        
        # Save config if available
        if self.config is not None:
            checkpoint['config'] = self.config
        
        if is_best:
            filename = "best.pt"
            self.best_val_f1 = val_metrics.get('f1', 0.0)
        else:
            filename = f"checkpoint_epoch_{epoch + 1}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if self.logger:
            self.logger.save_checkpoint(
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                epoch=epoch,
                metrics=val_metrics,
                is_best=is_best
            )
    
    def validate(self, evaluator: Optional[Any] = None, log_visualizations: bool = False) -> Dict[str, float]:
        """
        Validate the DETR model.
        
        Overrides base validate to handle DETR outputs and post-processing.
        
        Args:
            evaluator: Optional evaluator for computing metrics (TableEvaluator for backward compatibility)
            log_visualizations: Whether to collect per-image statistics for bucket-based visualization
        
        Returns:
            Dictionary with aggregated metrics. If log_visualizations=True, also includes
            'per_image_data' key with list of per-image statistics and batch data.
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        all_gt_boxes = []
        all_pred_boxes = []
        
        # For bucket-based visualization: collect per-image data
        per_image_data = []
        hungarian_evaluator = None
        
        if log_visualizations:
            hungarian_evaluator = HungarianEvaluator(use_giou=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                if isinstance(batch, dict):
                    features = batch['features'].to(self.device)
                    # Add batch dimension if missing (batch_size=1 case)
                    if features.dim() == 3:
                        features = features.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
                    
                    mask = batch.get('mask')
                    targets = batch.get('targets')
                    gt_boxes = batch.get('gt_boxes', [])
                    # Get dimensions from batch metadata or features
                    if features.dim() == 4:
                        sheet_height = batch.get('sheet_height', features.shape[2])
                        sheet_width = batch.get('sheet_width', features.shape[3])
                    else:
                        sheet_height = batch.get('sheet_height', features.shape[1])
                        sheet_width = batch.get('sheet_width', features.shape[2])
                    
                    # Mask is None for variable-sized sheets (all positions valid)
                    # If mask is provided, ensure it has batch dimension
                    if mask is not None:
                        mask = mask.to(self.device)
                        if mask.dim() == 2:
                            mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
                    
                    # Forward pass (inference mode)
                    model_output = self.model(x=features, mask=mask, targets=None)
                    
                    # Compute loss if targets available
                    if targets is not None:
                        targets_device = [{
                            'labels': targets['labels'].to(self.device),
                            'boxes': targets['boxes'].to(self.device),
                        }]
                        loss_output = self.model(x=features, mask=mask, targets=targets_device)
                        if isinstance(loss_output, dict) and 'losses' in loss_output:
                            losses = loss_output['losses']
                            if isinstance(losses, dict):
                                val_loss += sum(losses.values()).item()
                            else:
                                val_loss += losses.item()
                        num_batches += 1
                    
                    # Post-process predictions to boxes
                    pred_boxes_list = []
                    if isinstance(model_output, dict):
                        pred_logits = model_output.get('pred_logits')  # (B, num_queries, num_classes)
                        pred_boxes = model_output.get('pred_boxes')  # (B, num_queries, 4) normalized (cx, cy, w, h)
                        
                        if pred_logits is not None and pred_boxes is not None:
                            # Apply softmax to get class probabilities
                            pred_probs = torch.softmax(pred_logits, dim=-1)  # (B, num_queries, num_classes)
                            pred_scores = pred_probs[0, :, 1]  # Class 1 (table) probabilities
                            
                            # Filter by confidence threshold
                            confidence_threshold = 0.5
                            valid_indices = pred_scores > confidence_threshold
                            
                            if valid_indices.any():
                                valid_boxes = pred_boxes[0, valid_indices]  # (N, 4)
                                
                                # Convert normalized boxes to cell coordinates
                                pred_boxes_list = _convert_normalized_boxes_to_cells(
                                    valid_boxes.cpu().numpy(),
                                    sheet_height,
                                    sheet_width,
                                )
                    
                    all_pred_boxes.append(pred_boxes_list)
                    all_gt_boxes.append(gt_boxes)
                    
                    # Collect per-image statistics for bucket-based visualization
                    if log_visualizations and hungarian_evaluator is not None:
                        stats = hungarian_evaluator.evaluate_image(
                            gt_boxes=gt_boxes,
                            pred_boxes=pred_boxes_list,
                            sheet_width=sheet_width,
                            sheet_height=sheet_height,
                        )
                        
                        per_image_data.append({
                            'stats': stats,
                            'file_name': batch.get('file_name', 'unknown'),
                            'sheet_name': batch.get('sheet_name', 'unknown'),
                            'batch_data': {
                                'features': features,
                                'gt_boxes': gt_boxes,
                                'pred_boxes': pred_boxes_list,
                                'sheet_height': sheet_height,
                                'sheet_width': sheet_width,
                            },
                        })
        
        metrics = {}
        if num_batches > 0:
            metrics['loss'] = val_loss / num_batches
        
        # Run evaluator if provided (for backward compatibility)
        if evaluator is not None and len(all_gt_boxes) > 0:
            eval_results = evaluator.evaluate_batch(all_gt_boxes, all_pred_boxes)
            metrics.update(eval_results)
        
        # Add per-image data for bucket selection
        if log_visualizations:
            metrics['per_image_data'] = per_image_data
        
        return metrics


def log_prediction_image(
    sheet_features: torch.Tensor,
    gt_boxes: List[Tuple[int, int, int, int]],
    pred_boxes: List[Tuple[int, int, int, int]],
    epoch: int,
    sheet_name: str,
    file_name: str,
    matched_pairs: Optional[List[Tuple[int, int, float, float]]] = None,
    unmatched_gt_indices: Optional[List[int]] = None,
    unmatched_pred_indices: Optional[List[int]] = None,
    metrics_dict: Optional[Dict[str, float]] = None,
    bucket: Optional[str] = None,
    global_step: Optional[int] = None,
) -> None:
    """
    Log a visualization comparing ground truth vs predicted boxes with bucket-based enhancements.
    
    Args:
        sheet_features: Feature tensor (C, H, W) or (1, C, H, W)
        gt_boxes: List of (col_left, row_top, col_right, row_bottom) tuples
        pred_boxes: List of (col_left, row_top, col_right, row_bottom) tuples
        epoch: Current epoch number
        sheet_name: Name of the sheet for labeling
        file_name: Name of the file for labeling
        matched_pairs: List of (gt_idx, pred_idx, iou, nL1) tuples for matched pairs
        unmatched_gt_indices: List of unmatched GT box indices (misses)
        unmatched_pred_indices: List of unmatched prediction indices (extras)
        metrics_dict: Dictionary with metrics (T, P, M, miss_rate, extra_rate, mean_iou, p10_iou, mean_nL1)
        bucket: Bucket name ('great', 'bad', 'high_miss', 'high_extra')
        global_step: Global step number for logging
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert features to displayable format (use first channel or mean)
        if isinstance(sheet_features, torch.Tensor):
            if sheet_features.dim() == 4:
                # (B, C, H, W) -> take first batch, first channel
                display_img = sheet_features[0, 0].cpu().numpy()
            elif sheet_features.dim() == 3:
                # (C, H, W) -> use first channel
                display_img = sheet_features[0].cpu().numpy()
            else:
                display_img = sheet_features.mean(dim=0).cpu().numpy()
        else:
            display_img = sheet_features[0] if len(sheet_features.shape) == 3 else sheet_features.mean(axis=0)
        
        ax.imshow(display_img, cmap='gray', aspect='auto')
        
        # Track legend entries
        legend_handles = []
        legend_labels = []
        
        # Default values if not provided
        if matched_pairs is None:
            matched_pairs = []
        if unmatched_gt_indices is None:
            unmatched_gt_indices = []
        if unmatched_pred_indices is None:
            unmatched_pred_indices = []
        if metrics_dict is None:
            metrics_dict = {}
        
        # Draw matched pairs with yellow fill/hatch overlay
        matched_label_added = False
        for gt_idx, pred_idx, iou, nL1 in matched_pairs:
            gt_box = gt_boxes[gt_idx]
            pred_box = pred_boxes[pred_idx]
            col_left, row_top, col_right, row_bottom = gt_box
            width = col_right - col_left
            height = row_bottom - row_top
            
            # Draw GT box (green, solid)
            label = f'Matched GT: IoU={iou:.3f}, nL1={nL1:.4f}' if not matched_label_added else None
            rect_gt = plt.Rectangle(
                (col_left - 1, row_top - 1), width, height,
                fill=False, edgecolor='green', linewidth=2, label=label
            )
            ax.add_patch(rect_gt)
            if label:
                legend_handles.append(rect_gt)
                legend_labels.append(label)
                matched_label_added = True
            
            # Draw pred box (red, dashed)
            pred_col_left, pred_row_top, pred_col_right, pred_row_bottom = pred_box
            pred_width = pred_col_right - pred_col_left
            pred_height = pred_row_bottom - pred_row_top
            rect_pred = plt.Rectangle(
                (pred_col_left - 1, pred_row_top - 1), pred_width, pred_height,
                fill=False, edgecolor='red', linewidth=2, linestyle='--'
            )
            ax.add_patch(rect_pred)
            
            # Draw yellow fill overlay for matched overlap
            overlap_left = max(col_left, pred_col_left)
            overlap_top = max(row_top, pred_row_top)
            overlap_right = min(col_right, pred_col_right)
            overlap_bottom = min(row_bottom, pred_row_bottom)
            if overlap_left < overlap_right and overlap_top < overlap_bottom:
                overlap_width = overlap_right - overlap_left
                overlap_height = overlap_bottom - overlap_top
                rect_overlap = plt.Rectangle(
                    (overlap_left - 1, overlap_top - 1), overlap_width, overlap_height,
                    fill=True, facecolor='yellow', alpha=0.3, edgecolor='none'
                )
                ax.add_patch(rect_overlap)
        
        # Draw unmatched GT boxes (blue outline - misses)
        miss_label_added = False
        for gt_idx in unmatched_gt_indices:
            gt_box = gt_boxes[gt_idx]
            col_left, row_top, col_right, row_bottom = gt_box
            width = col_right - col_left
            height = row_bottom - row_top
            coord_str = f"({col_left}, {row_top}, {col_right}, {row_bottom})"
            label = f'Missed GT: {coord_str}' if not miss_label_added else None
            rect = plt.Rectangle(
                (col_left - 1, row_top - 1), width, height,
                fill=False, edgecolor='blue', linewidth=2, linestyle=':', label=label
            )
            ax.add_patch(rect)
            if label:
                legend_handles.append(rect)
                legend_labels.append(label)
                miss_label_added = True
        
        # Draw unmatched pred boxes (orange outline - extras)
        extra_label_added = False
        for pred_idx in unmatched_pred_indices:
            pred_box = pred_boxes[pred_idx]
            col_left, row_top, col_right, row_bottom = pred_box
            width = col_right - col_left
            height = row_bottom - row_top
            coord_str = f"({col_left}, {row_top}, {col_right}, {row_bottom})"
            label = f'Extra Pred: {coord_str}' if not extra_label_added else None
            rect = plt.Rectangle(
                (col_left - 1, row_top - 1), width, height,
                fill=False, edgecolor='orange', linewidth=2, linestyle='-.', label=label
            )
            ax.add_patch(rect)
            if label:
                legend_handles.append(rect)
                legend_labels.append(label)
                extra_label_added = True
        
        # Add title with bucket info
        title = f'Epoch {epoch}: {file_name} / {sheet_name}'
        if bucket:
            title += f' [{bucket.upper()}]'
        ax.set_title(title, fontsize=10)
        
        # Add legend
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=7)
        
        # Add caption block with metrics
        if metrics_dict:
            caption_lines = [
                f'Epoch: {epoch}',
            ]
            if global_step is not None:
                caption_lines.append(f'Step: {global_step}')
            caption_lines.append(f'Image: {file_name} / {sheet_name}')
            
            T = metrics_dict.get('T', 0)
            P = metrics_dict.get('P', 0)
            M = metrics_dict.get('M', 0)
            caption_lines.append(f'T={T}, P={P}, M={M}')
            
            miss_rate = metrics_dict.get('miss_rate', 0.0)
            extra_rate = metrics_dict.get('extra_rate', 0.0)
            caption_lines.append(f'miss_rate={miss_rate:.3f}, extra_rate={extra_rate:.3f}')
            
            mean_iou = metrics_dict.get('mean_iou', 0.0)
            p10_iou = metrics_dict.get('p10_iou', 0.0)
            mean_nL1 = metrics_dict.get('mean_nL1', 0.0)
            caption_lines.append(f'mean_iou={mean_iou:.3f}, p10_iou={p10_iou:.3f}')
            caption_lines.append(f'mean_nL1={mean_nL1:.4f}')
            
            # Top-3 matched pairs
            if matched_pairs:
                top_pairs = sorted(matched_pairs, key=lambda x: x[2], reverse=True)[:3]
                pair_strs = []
                for gt_idx, pred_idx, iou, nL1 in top_pairs:
                    pair_strs.append(f'GT{gt_idx}-Pred{pred_idx}: IoU={iou:.3f}, nL1={nL1:.4f}')
                caption_lines.append('Top-3: ' + '; '.join(pair_strs))
            
            caption_text = '\n'.join(caption_lines)
            ax.text(
                0.02, 0.98, caption_text,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace'
            )
        
        ax.axis('off')
        
        # Sanitize sheet name for W&B
        sanitized_sheet_name = "".join(c if c.isalnum() or c in '-_.' else '_' for c in sheet_name)
        bucket_prefix = f"{bucket}/" if bucket else ""
        wandb_key = f"val/{bucket_prefix}predictions_{sanitized_sheet_name}"
        wandb.log({wandb_key: wandb.Image(fig)}, step=epoch if global_step is None else global_step)
        plt.close(fig)
    except Exception as e:
        # Don't fail training if visualization fails
        import warnings
        warnings.warn(f"Failed to log prediction image: {e}")
        plt.close('all')


def _convert_normalized_boxes_to_cells(
    boxes_normalized: np.ndarray,
    sheet_height: int,
    sheet_width: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Convert normalized (cx, cy, w, h) boxes to cell-aligned (col_left, row_top, col_right, row_bottom).
    
    Args:
        boxes_normalized: Array of shape (N, 4) with normalized (cx, cy, w, h)
        sheet_height: Sheet height in cells
        sheet_width: Sheet width in cells
        
    Returns:
        List of tuples (col_left, row_top, col_right, row_bottom) in 1-based indices
    """
    boxes_cells = []
    
    for box in boxes_normalized:
        cx, cy, w, h = box
        
        # Convert to (x1, y1, x2, y2) in normalized coordinates
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        # Clamp to [0, 1]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        
        # Convert to cell coordinates (1-based)
        col_left = int(np.floor(x1 * sheet_width)) + 1
        row_top = int(np.floor(y1 * sheet_height)) + 1
        col_right = int(np.ceil(x2 * sheet_width))
        row_bottom = int(np.ceil(y2 * sheet_height))
        
        # Ensure valid bounds
        col_left = max(1, min(col_left, sheet_width))
        row_top = max(1, min(row_top, sheet_height))
        col_right = max(1, min(col_right, sheet_width))
        row_bottom = max(1, min(row_bottom, sheet_height))
        
        # Skip degenerate boxes
        if col_left < col_right and row_top < row_bottom:
            boxes_cells.append((col_left, row_top, col_right, row_bottom))
    
    return boxes_cells


def create_data_loaders(
    project_root: Path,
    dataset_names: list,
    batch_size: int = 1,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
):
    """
    Create data loaders for training and validation.
    
    Args:
        project_root: Root directory of the project
        dataset_names: List of dataset names to load
        batch_size: Batch size (default: 1 for variable sheet sizes)
        num_workers: Number of data loader workers
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        split_seed: Seed for deterministic splitting
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load annotations
    loader = AnnotationLoader(
        project_root=project_root,
        dataset_names=dataset_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
    )
    loader.load_annotations()
    
    # Get annotations by split
    train_annotations = loader.get_annotations_by_split('train')
    val_annotations = loader.get_annotations_by_split('val')
    
    print(f"Training samples: {len(train_annotations)}")
    print(f"Validation samples: {len(val_annotations)}")
    
    # Create featurizer
    featurizer = CellFeaturizer()
    
    # Create datasets
    train_dataset = DETRDataset(train_annotations, project_root, featurizer)
    val_dataset = DETRDataset(val_annotations, project_root, featurizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=lambda batch: batch[0],  # Since batch_size=1, just return the single item
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=lambda batch: batch[0],
    )
    
    return train_loader, val_loader


def _merge_sweep_config(base_config: dict, sweep_config: dict) -> dict:
    """
    Merge W&B sweep config into base config with nested key support.
    Example: sweep key 'training.lr' updates base_config['training']['lr']
    
    Args:
        base_config: Base configuration dictionary
        sweep_config: Sweep configuration dictionary from W&B
        
    Returns:
        Merged configuration dictionary
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


def create_detector_from_config(config: dict) -> TableDetector:
    """
    Create TableDetector model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TableDetector instance
    """
    model_config = config['model']
    backbone_config = model_config.get('backbone', {})
    detr_config = model_config.get('detr', {})
    loss_config = model_config.get('loss', {})
    
    # Get feature count (43 features per requirements v1.2, section 5.1.3)
    in_channels = 43  # From requirements v1.2: 43 features
    
    detector = TableDetector(
        in_channels=in_channels,
        backbone_base_width=backbone_config.get('base_width', 64),
        backbone_depths=backbone_config.get('depths', [3, 3, 9, 3]),
        num_queries=detr_config.get('num_queries', 20),
        num_encoder_layers=detr_config.get('num_encoder_layers', 6),
        num_decoder_layers=detr_config.get('num_decoder_layers', 6),
        hidden_dim=detr_config.get('hidden_dim', 256),
        dropout=detr_config.get('dropout', 0.1),
        loss_weights={
            'loss_ce': loss_config.get('lambda_cls', 1.0),
            'loss_bbox': loss_config.get('lambda_l1', 5.0),
            'loss_giou': loss_config.get('lambda_iou', 2.0),
        },
    )
    
    return detector


def main():
    parser = argparse.ArgumentParser(description="Train Deformable DETR table detector")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run as part of W&B sweep"
    )
    
    args = parser.parse_args()
    
    # 1. Load Base Config
    cfg = Config.from_yaml(args.config)
    config = cfg.to_dict()
    
    # 2. Merge Sweep Config if active
    if args.sweep:
        # In sweep mode, wandb.init() is called by the sweep agent before our script runs
        # We just need to access wandb.config to get the sweep parameters
        import wandb
        # Ensure wandb is initialized (idempotent if already initialized)
        wandb.init(project=config['experiment'].get('project_name', 'tablesense2'))
        # Merge sweep config into base config
        config = _merge_sweep_config(config, dict(wandb.config))
        wandb_run_id = wandb.run.id if wandb.run else None
    else:
        wandb_run_id = None
    
    # Setup paths
    project_root = Path(args.project_root).resolve()
    
    # Create data loaders
    # Ensure numeric values are properly typed (YAML might read them as strings)
    train_loader, val_loader = create_data_loaders(
        project_root=project_root,
        dataset_names=config['data']['dataset_names'],
        batch_size=int(config['training'].get('batch_size', 1)),
        num_workers=int(config['training'].get('num_workers', 0)),
        train_ratio=float(config['data'].get('train_ratio', 0.8)),
        val_ratio=float(config['data'].get('val_ratio', 0.1)),
        test_ratio=float(config['data'].get('test_ratio', 0.1)),
        split_seed=int(config['data'].get('split_seed', 42)),
    )
    
    # Create model
    model = create_detector_from_config(config)
    
    # 3. Initialize Logger (modified to handle existing run)
    if args.sweep:
        # If sweep, we already initialized wandb above.
        # Pass existing config and rely on global wandb state
        logger = ExperimentLogger(
            experiment_name=config['experiment']['name'],
            project_name=config['experiment'].get('project_name', 'tablesense2'),
            config=config,
            resume=wandb_run_id,
        )
    else:
        # Normal run
        logger = ExperimentLogger(
            experiment_name=config['experiment']['name'],
            project_name=config['experiment'].get('project_name', 'tablesense2'),
            config=config,
        )
    
    # 4. Enhanced Logging
    logger.log_model_summary(model)
    logger.log_dataset_stats(
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        dataset_names=config['data']['dataset_names']
    )
    
    # Log hyperparameters explicitly for sweeps
    logger.log_hyperparameters({
        'lr': config['training'].get('lr', 1e-4),
        'weight_decay': config['training'].get('weight_decay', 1e-4),
        'batch_size': config['training'].get('batch_size', 1),
        'gradient_accumulation_steps': config['training'].get('gradient_accumulation_steps', 8),
        'warmup_epochs': config['training'].get('warmup_epochs', 3),
        'num_queries': config['model']['detr'].get('num_queries', 20),
        'hidden_dim': config['model']['detr'].get('hidden_dim', 256),
        'num_encoder_layers': config['model']['detr'].get('num_encoder_layers', 6),
        'num_decoder_layers': config['model']['detr'].get('num_decoder_layers', 6),
        'dropout': config['model']['detr'].get('dropout', 0.1),
        'lambda_cls': config['model']['loss'].get('lambda_cls', 1.0),
        'lambda_l1': config['model']['loss'].get('lambda_l1', 5.0),
        'lambda_iou': config['model']['loss'].get('lambda_iou', 2.0),
    })
    
    # Create evaluator (will be used in validation)
    from evaluation.evaluator import TableEvaluator
    evaluator = TableEvaluator(eob_threshold=float(config['evaluation'].get('eob_threshold', 2.0)))
    
    # Create trainer
    # Ensure numeric values are properly typed (YAML might read them as strings)
    trainer = DETRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        config=config,  # Pass config for saving in checkpoints
        device=config['training'].get('device', _get_default_device()),
        gradient_accumulation_steps=int(config['training'].get('gradient_accumulation_steps', 8)),
        max_epochs=int(config['training'].get('max_epochs', 100)),
        lr=float(config['training'].get('lr', 1e-4)),
        weight_decay=float(config['training'].get('weight_decay', 1e-4)),
        warmup_epochs=int(config['training'].get('warmup_epochs', 3)),
        checkpoint_dir=logger.checkpoint_dir,
        save_every_n_epochs=int(config['training'].get('save_every_n_epochs', 10)),
        log_gradient_norms=True,  # Enable gradient norm logging for training stability monitoring
    )
    
    # Train
    trainer.train(
        evaluator=evaluator,
        early_stopping_patience=config['training'].get('early_stopping_patience', None)
    )
    
    logger.finish()
    print("Training completed!")


if __name__ == "__main__":
    main()

