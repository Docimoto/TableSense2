"""
Evaluation script for Deformable DETR table detector.

Supports three evaluation modes:
- CNN-only: Evaluate DETR predictions
- Excel-only: Evaluate Excel table objects
- Hybrid: Evaluate aggregated predictions (DETR + Excel)
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.table_detector import TableDetector
from scripts.train_detector import DETRDataset, create_detector_from_config, _convert_normalized_boxes_to_cells
from data_io.annotations import AnnotationLoader
from features.featurizer import CellFeaturizer
from evaluation.evaluator import TableEvaluator
from evaluation.metrics import compute_iou
from utils.excel_utils import load_workbook, parse_excel_range
from utils.detect_excel_tables import get_table_regions


def post_process_detr_outputs(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    sheet_height: int,
    sheet_width: int,
    confidence_threshold: float = 0.5,
    nms_iou_threshold: float = 0.3,
    max_detections: int = 50,
) -> List[Tuple[int, int, int, int]]:
    """
    Post-process DETR outputs to get final detections.
    
    Args:
        pred_logits: (num_queries, num_classes) classification logits
        pred_boxes: (num_queries, 4) normalized boxes (cx, cy, w, h)
        sheet_height: Sheet height in cells
        sheet_width: Sheet width in cells
        confidence_threshold: Confidence threshold for filtering
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections
        
    Returns:
        List of boxes as (col_left, row_top, col_right, row_bottom) in 1-based indices
    """
    # Apply softmax to get class probabilities
    pred_probs = torch.softmax(pred_logits, dim=-1)  # (num_queries, num_classes)
    pred_scores = pred_probs[:, 1].cpu().numpy()  # Class 1 (table) probabilities
    
    # Filter by confidence threshold
    valid_indices = np.where(pred_scores > confidence_threshold)[0]
    
    if len(valid_indices) == 0:
        return []
    
    valid_boxes = pred_boxes[valid_indices].cpu().numpy()  # (N, 4)
    valid_scores = pred_scores[valid_indices]
    
    # Convert normalized boxes to cell coordinates
    boxes_cells = _convert_normalized_boxes_to_cells(
        valid_boxes,
        sheet_height,
        sheet_width,
    )
    
    # Apply NMS (Non-Maximum Suppression)
    if len(boxes_cells) > 1:
        boxes_cells, valid_scores = apply_nms(
            boxes_cells,
            valid_scores,
            iou_threshold=nms_iou_threshold,
        )
    
    # Sort by score and take top max_detections
    if len(boxes_cells) > max_detections:
        sorted_indices = np.argsort(valid_scores)[::-1][:max_detections]
        boxes_cells = [boxes_cells[i] for i in sorted_indices]
    
    return boxes_cells


def apply_nms(
    boxes: List[Tuple[int, int, int, int]],
    scores: np.ndarray,
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: List of boxes as (col_left, row_top, col_right, row_bottom)
        scores: Array of confidence scores
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Tuple of (filtered_boxes, filtered_scores)
    """
    if len(boxes) == 0:
        return boxes, scores
    
    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    boxes = [boxes[i] for i in sorted_indices]
    scores = scores[sorted_indices]
    
    # Compute IoU matrix
    n = len(boxes)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                iou_matrix[i, j] = compute_iou(boxes[i], boxes[j])
    
    # Greedy NMS
    keep = []
    suppressed = set()
    
    for i in range(n):
        if i in suppressed:
            continue
        
        keep.append(i)
        
        # Suppress boxes with high IoU
        for j in range(i + 1, n):
            if j not in suppressed and iou_matrix[i, j] > iou_threshold:
                suppressed.add(j)
    
    filtered_boxes = [boxes[i] for i in keep]
    filtered_scores = scores[keep]
    
    return filtered_boxes, filtered_scores


def detect_excel_tables(
    workbook_path: Path,
    sheet_name: str,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect Excel table objects in a worksheet.
    
    Args:
        workbook_path: Path to Excel workbook
        sheet_name: Name of the sheet
        
    Returns:
        List of boxes as (col_left, row_top, col_right, row_bottom) in 1-based indices
    """
    workbook = load_workbook(workbook_path, data_only=False, read_only=True)
    worksheet = workbook[sheet_name]
    
    table_regions = get_table_regions(worksheet)
    
    boxes = []
    for region_str in table_regions:
        try:
            col_left, row_top, col_right, row_bottom = parse_excel_range(region_str)
            boxes.append((col_left, row_top, col_right, row_bottom))
        except ValueError:
            # Skip invalid ranges
            continue
    
    workbook.close()
    return boxes


def aggregate_hybrid(
    cnn_boxes: List[Tuple[int, int, int, int]],
    excel_boxes: List[Tuple[int, int, int, int]],
    tau_merge: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """
    Aggregate CNN and Excel detections using IoU-based merging.
    
    Args:
        cnn_boxes: List of CNN-detected boxes
        excel_boxes: List of Excel-detected boxes
        tau_merge: IoU threshold for merging boxes
        
    Returns:
        List of aggregated boxes
    """
    if len(cnn_boxes) == 0:
        return excel_boxes
    if len(excel_boxes) == 0:
        return cnn_boxes
    
    # Start with all CNN boxes
    aggregated = list(cnn_boxes)
    matched_excel = set()
    
    # Try to match Excel boxes with CNN boxes
    for excel_box in excel_boxes:
        best_iou = 0.0
        best_match_idx = None
        
        for idx, cnn_box in enumerate(aggregated):
            iou = compute_iou(excel_box, cnn_box)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = idx
        
        if best_iou >= tau_merge:
            # Merge: take union of boxes
            matched_excel.add(excel_box)
            if best_match_idx is not None:
                cnn_box = aggregated[best_match_idx]
                # Union: min of left/top, max of right/bottom
                merged_box = (
                    min(cnn_box[0], excel_box[0]),
                    min(cnn_box[1], excel_box[1]),
                    max(cnn_box[2], excel_box[2]),
                    max(cnn_box[3], excel_box[3]),
                )
                aggregated[best_match_idx] = merged_box
        else:
            # No match: add Excel box
            aggregated.append(excel_box)
    
    return aggregated


def evaluate_model(
    model_path: Path,
    project_root: Path,
    dataset_names: list,
    split: str = 'val',
    batch_size: int = 1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    eob_threshold: float = 2.0,
    confidence_threshold: float = 0.5,
    nms_iou_threshold: float = 0.3,
    max_detections: int = 50,
    hybrid_tau_merge: float = 0.5,
    mode: str = 'cnn_only',  # 'cnn_only', 'excel_only', 'hybrid'
) -> Dict[str, Any]:
    """
    Evaluate a trained detector model.
    
    Args:
        model_path: Path to model checkpoint
        project_root: Root directory of the project
        dataset_names: List of dataset names to evaluate on
        split: Split to evaluate on ('train', 'val', 'test')
        batch_size: Batch size
        device: Device to run evaluation on
        eob_threshold: EoB threshold for evaluation
        confidence_threshold: Confidence threshold for predictions
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections per sheet
        hybrid_tau_merge: IoU threshold for hybrid aggregation
        mode: Evaluation mode ('cnn_only', 'excel_only', 'hybrid')
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model - try to load config from checkpoint, otherwise use provided config
    if 'config' in checkpoint:
        # Use config from checkpoint
        model_config = checkpoint['config']
    else:
        # Use provided config file
        model_config = config
    
    model = create_detector_from_config(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load annotations
    loader = AnnotationLoader(
        project_root=project_root,
        dataset_names=dataset_names,
    )
    loader.load_annotations()
    
    # Get annotations for split
    if split == 'train':
        annotations = loader.get_annotations_by_split('train')
    elif split == 'val':
        annotations = loader.get_annotations_by_split('val')
    elif split == 'test':
        annotations = loader.get_annotations_by_split('test')
    else:
        raise ValueError(f"Invalid split: {split}")
    
    print(f"Evaluating on {split} split: {len(annotations)} samples")
    print(f"Mode: {mode}")
    
    # Create dataset and loader
    featurizer = CellFeaturizer()
    dataset = DETRDataset(annotations, project_root, featurizer)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch[0],
    )
    
    # Create evaluator
    evaluator = TableEvaluator(eob_threshold=eob_threshold)
    
    # Evaluate
    all_gt_boxes = []
    all_pred_boxes = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            features = batch['features'].to(device)
            mask = batch.get('mask')
            gt_boxes = batch['gt_boxes']
            workbook_id = batch['workbook_id']
            sheet_name = batch['sheet_name']
            sheet_height = batch.get('sheet_height', features.shape[2])
            sheet_width = batch.get('sheet_width', features.shape[3])
            
            if mask is not None:
                mask = mask.to(device)
            
            pred_boxes_list = []
            
            if mode == 'cnn_only' or mode == 'hybrid':
                # Forward pass
                model_output = model(x=features, mask=mask, targets=None)
                
                if isinstance(model_output, dict):
                    pred_logits = model_output.get('pred_logits')  # (B, num_queries, num_classes)
                    pred_boxes = model_output.get('pred_boxes')  # (B, num_queries, 4)
                    
                    if pred_logits is not None and pred_boxes is not None:
                        # Post-process
                        cnn_boxes = post_process_detr_outputs(
                            pred_logits[0],  # First (and only) batch item
                            pred_boxes[0],
                            sheet_height,
                            sheet_width,
                            confidence_threshold=confidence_threshold,
                            nms_iou_threshold=nms_iou_threshold,
                            max_detections=max_detections,
                        )
                        
                        if mode == 'cnn_only':
                            pred_boxes_list = cnn_boxes
                        else:
                            # Hybrid mode: also get Excel boxes
                            file_path = project_root / batch.get('file_path', '') / batch.get('file_name', '')
                            if file_path.exists() and file_path.is_file():
                                excel_boxes = detect_excel_tables(file_path, sheet_name)
                                pred_boxes_list = aggregate_hybrid(cnn_boxes, excel_boxes, tau_merge=hybrid_tau_merge)
                            else:
                                pred_boxes_list = cnn_boxes
            
            elif mode == 'excel_only':
                # Excel-only mode
                file_path = project_root / batch.get('file_path', '') / batch.get('file_name', '')
                if file_path.exists() and file_path.is_file():
                    pred_boxes_list = detect_excel_tables(file_path, sheet_name)
                else:
                    pred_boxes_list = []
            
            all_gt_boxes.append(gt_boxes)
            all_pred_boxes.append(pred_boxes_list)
    
    # Compute metrics
    results = evaluator.evaluate_batch(all_gt_boxes, all_pred_boxes)
    
    # Add metadata
    results['mode'] = mode
    results['split'] = split
    results['num_samples'] = len(annotations)
    results['config'] = {
        'confidence_threshold': confidence_threshold,
        'nms_iou_threshold': nms_iou_threshold,
        'max_detections': max_detections,
        'eob_threshold': eob_threshold,
        'hybrid_tau_merge': hybrid_tau_merge if mode == 'hybrid' else None,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Deformable DETR table detector")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=['train', 'val', 'test'],
        help="Split to evaluate on (default: val)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cnn_only",
        choices=['cnn_only', 'excel_only', 'hybrid'],
        help="Evaluation mode (default: cnn_only)"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON (optional)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5)"
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for NMS (default: 0.3)"
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=50,
        help="Maximum number of detections per sheet (default: 50)"
    )
    parser.add_argument(
        "--hybrid-tau-merge",
        type=float,
        default=0.5,
        help="IoU threshold for hybrid aggregation (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model).resolve()
    
    # Evaluate
    results = evaluate_model(
        model_path=model_path,
        project_root=project_root,
        dataset_names=config['data']['dataset_names'],
        split=args.split,
        batch_size=config['training'].get('batch_size', 1),
        device=config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        eob_threshold=config['evaluation'].get('eob_threshold', 2.0),
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
        hybrid_tau_merge=args.hybrid_tau_merge,
        mode=args.mode,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({results['mode']} mode)")
    print("=" * 50)
    print(f"Split: {results['split']}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"True Positives: {results['tp']}")
    print(f"False Positives: {results['fp']}")
    print(f"False Negatives: {results['fn']}")
    print(f"Mean EoB: {results['eob_mean']:.4f}")
    print(f"Std EoB: {results['eob_std']:.4f}")
    print("=" * 50)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        # Convert numpy types to Python types for JSON serialization
        results_json = {}
        for k, v in results.items():
            if k == 'config':
                results_json[k] = v
            elif isinstance(v, (np.integer, np.floating)):
                results_json[k] = float(v)
            elif isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            else:
                results_json[k] = v
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

