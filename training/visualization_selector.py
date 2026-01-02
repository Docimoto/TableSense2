"""
Visualization selector for bucket-based image selection.

Implements bucket classification (Great/Bad/High Miss/High Extra),
badness scoring, ranking, and cache management to avoid duplicate visualizations.
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np


class VisualizationSelector:
    """
    Selects images for visualization based on performance buckets.
    
    Classifies images into buckets (Great, Bad, High Miss, High Extra),
    ranks them using a "badness" score, and manages a cache to avoid
    printing the same images repeatedly.
    """
    
    # Bucket thresholds from document
    GREAT_THRESHOLDS = {
        'miss_rate': 0.0,
        'extra_rate': 0.25,
        'mean_iou': 0.85,
        'p10_iou': 0.75,
        'mean_nL1': 0.020,
    }
    
    BAD_THRESHOLDS = {
        'miss_rate': 0.34,
        'mean_iou': 0.35,
        'p10_iou': 0.20,
        'mean_nL1': 0.080,
    }
    
    HIGH_MISS_THRESHOLDS = {
        'miss_rate': 0.50,
        'U_t': 1,
        'mean_iou': 0.60,
    }
    
    HIGH_EXTRA_THRESHOLDS = {
        'extra_rate': 1.0,
        'U_p': 2,
    }
    
    def __init__(
        self,
        best_count: int = 20,
        worst_count: int = 20,
        high_miss_count: int = 20,
        high_extra_count: int = 20,
        cache_epochs: int = 3,
    ):
        """
        Initialize visualization selector.
        
        Args:
            best_count: Number of "Great" images to select
            worst_count: Number of "Bad" images to select
            high_miss_count: Number of "High Miss" images to select
            high_extra_count: Number of "High Extra" images to select
            cache_epochs: Number of epochs to track in cache (default: 3)
        """
        self.best_count = best_count
        self.worst_count = worst_count
        self.high_miss_count = high_miss_count
        self.high_extra_count = high_extra_count
        self.cache_epochs = cache_epochs
        
        # Cache: maps (file_name, sheet_name) -> set of epochs when printed
        self.recent_cache: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    
    def compute_badness_score(self, stats: Dict[str, float]) -> float:
        """
        Compute "badness" score for ranking (lower is better).
        
        Formula: (1 - mean_iou) + 0.70 * miss_rate + 0.35 * min(extra_rate, 2.0) + 0.25 * min(mean_nL1 / 0.05, 2.0)
        
        Args:
            stats: Per-image statistics dictionary
            
        Returns:
            Badness score (lower is better)
        """
        mean_iou = stats.get('mean_iou', 0.0)
        miss_rate = stats.get('miss_rate', 0.0)
        extra_rate = stats.get('extra_rate', 0.0)
        mean_nL1 = stats.get('mean_nL1', 1000.0)
        
        badness = (
            (1.0 - mean_iou) +
            0.70 * miss_rate +
            0.35 * min(extra_rate, 2.0) +
            0.25 * min(mean_nL1 / 0.05, 2.0)
        )
        
        return badness
    
    def classify_bucket(self, stats: Dict[str, float]) -> Optional[str]:
        """
        Classify an image into a bucket based on thresholds.
        
        Args:
            stats: Per-image statistics dictionary
            
        Returns:
            Bucket name ('great', 'bad', 'high_miss', 'high_extra') or None
        """
        miss_rate = stats.get('miss_rate', 0.0)
        extra_rate = stats.get('extra_rate', 0.0)
        mean_iou = stats.get('mean_iou', 0.0)
        p10_iou = stats.get('p10_iou', 0.0)
        mean_nL1 = stats.get('mean_nL1', 1000.0)
        U_t = stats.get('U_t', 0)
        U_p = stats.get('U_p', 0)
        
        # Check High Miss first (most actionable)
        if miss_rate >= self.HIGH_MISS_THRESHOLDS['miss_rate']:
            return 'high_miss'
        if U_t >= self.HIGH_MISS_THRESHOLDS['U_t'] and mean_iou < self.HIGH_MISS_THRESHOLDS['mean_iou']:
            return 'high_miss'
        
        # Check High Extra
        if extra_rate >= self.HIGH_EXTRA_THRESHOLDS['extra_rate']:
            return 'high_extra'
        if U_p >= self.HIGH_EXTRA_THRESHOLDS['U_p']:
            return 'high_extra'
        
        # Check Bad
        if miss_rate >= self.BAD_THRESHOLDS['miss_rate']:
            return 'bad'
        if mean_iou > 0 and mean_iou <= self.BAD_THRESHOLDS['mean_iou']:
            return 'bad'
        if p10_iou > 0 and p10_iou <= self.BAD_THRESHOLDS['p10_iou']:
            return 'bad'
        if mean_nL1 >= self.BAD_THRESHOLDS['mean_nL1']:
            return 'bad'
        
        # Check Great
        if (miss_rate == self.GREAT_THRESHOLDS['miss_rate'] and
            extra_rate <= self.GREAT_THRESHOLDS['extra_rate'] and
            mean_iou >= self.GREAT_THRESHOLDS['mean_iou'] and
            p10_iou >= self.GREAT_THRESHOLDS['p10_iou'] and
            mean_nL1 <= self.GREAT_THRESHOLDS['mean_nL1']):
            return 'great'
        
        return None
    
    def select_images(
        self,
        image_data: List[Dict],
        current_epoch: int,
    ) -> Dict[str, List[Dict]]:
        """
        Select images from each bucket for visualization.
        
        Args:
            image_data: List of dictionaries, each containing:
                - 'stats': Per-image statistics from HungarianEvaluator
                - 'file_name': File name
                - 'sheet_name': Sheet name
                - 'batch_data': Full batch data for visualization
            current_epoch: Current epoch number
            
        Returns:
            Dictionary mapping bucket names to lists of selected image data
        """
        # Classify all images into buckets
        buckets: Dict[str, List[Tuple[Dict, float]]] = defaultdict(list)
        
        for img_data in image_data:
            stats = img_data['stats']
            bucket = self.classify_bucket(stats)
            
            if bucket is not None:
                badness = self.compute_badness_score(stats)
                buckets[bucket].append((img_data, badness))
        
        # Rank images within each bucket
        for bucket in buckets:
            # Sort by badness (ascending for great, descending for others)
            reverse = (bucket != 'great')
            buckets[bucket].sort(key=lambda x: x[1], reverse=reverse)
        
        # Select images from each bucket, respecting cache
        selected: Dict[str, List[Dict]] = {
            'great': [],
            'bad': [],
            'high_miss': [],
            'high_extra': [],
        }
        
        bucket_counts = {
            'great': self.best_count,
            'bad': self.worst_count,
            'high_miss': self.high_miss_count,
            'high_extra': self.high_extra_count,
        }
        
        # Clean old entries from cache
        min_epoch = current_epoch - self.cache_epochs
        for key in list(self.recent_cache.keys()):
            self.recent_cache[key] = {e for e in self.recent_cache[key] if e >= min_epoch}
            if not self.recent_cache[key]:
                del self.recent_cache[key]
        
        # Select from each bucket
        for bucket, count in bucket_counts.items():
            if bucket not in buckets:
                continue
            
            selected_count = 0
            for img_data, badness in buckets[bucket]:
                if selected_count >= count:
                    break
                
                file_name = img_data['file_name']
                sheet_name = img_data['sheet_name']
                cache_key = (file_name, sheet_name)
                
                # Skip if recently printed
                if cache_key in self.recent_cache:
                    continue
                
                selected[bucket].append(img_data)
                self.recent_cache[cache_key].add(current_epoch)
                selected_count += 1
        
        # Handle spillover: if a bucket has fewer candidates, fill from worst bucket
        total_selected = sum(len(selected[b]) for b in selected)
        total_target = sum(bucket_counts.values())
        
        if total_selected < total_target:
            # Collect remaining images from worst bucket (bad)
            remaining = []
            for img_data, badness in buckets['bad']:
                file_name = img_data['file_name']
                sheet_name = img_data['sheet_name']
                cache_key = (file_name, sheet_name)
                if cache_key not in self.recent_cache:
                    remaining.append((img_data, badness))
            
            # Sort by badness descending (worst first)
            remaining.sort(key=lambda x: x[1], reverse=True)
            
            # Fill up to target
            for img_data, badness in remaining:
                if total_selected >= total_target:
                    break
                
                file_name = img_data['file_name']
                sheet_name = img_data['sheet_name']
                cache_key = (file_name, sheet_name)
                
                if cache_key not in self.recent_cache:
                    selected['bad'].append(img_data)
                    self.recent_cache[cache_key].add(current_epoch)
                    total_selected += 1
        
        return selected
    
    def update_cache(self, current_epoch: int):
        """
        Clean old entries from cache.
        
        Args:
            current_epoch: Current epoch number
        """
        min_epoch = current_epoch - self.cache_epochs
        for key in list(self.recent_cache.keys()):
            self.recent_cache[key] = {e for e in self.recent_cache[key] if e >= min_epoch}
            if not self.recent_cache[key]:
                del self.recent_cache[key]
