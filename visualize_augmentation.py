#!/usr/bin/env python3
"""
Script to visualize augmentation results and store them in an augmentation folder.
Also includes a fixed threshold function for single keypoint detection.
"""

import os
import sys
import numpy as np
import cv2
import torch
import random
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from depth_keypoint_model_training import (
    generate_dataset_data, 
    KeypointDataset, 
    MinimalAugmentation, 
    StrongerAugmentation,
    RandomRotateFlip,
    ColorJitter,
    GaussianNoise,
    Cutout
)
from ultralytics import YOLO

def fixed_threshold_locations(heatmap, threshold=0.1):
    """
    Find single peak location in heatmap above threshold.
    
    Args:
        heatmap: (H, W) tensor or numpy array
        threshold: Threshold value
    
    Returns:
        List of (y, x) coordinates (single point per keypoint)
    """
    # Convert numpy array to tensor if needed
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    
    # Find regions above threshold
    mask = heatmap > threshold
    
    if not torch.any(mask):
        return []
    
    # Find connected components
    from scipy.ndimage import label
    mask_np = mask.numpy().astype(np.uint8)
    labeled, num_features = label(mask_np, structure=np.ones((3, 3), dtype=int))
    
    peaks = []
    for i in range(1, num_features + 1):
        # Get coordinates of this component
        coords = np.argwhere(labeled == i)
        
        if len(coords) == 0:
            continue
        
        # Find the peak (maximum value) within this component
        component_mask = (labeled == i)
        component_values = heatmap.numpy()[component_mask]
        component_coords = coords
        
        # Find the coordinate with maximum value
        max_idx = np.argmax(component_values)
        peak_coord = component_coords[max_idx]
        
        peaks.append((int(peak_coord[0]), int(peak_coord[1])))
    
    return peaks

def visualize_augmentation_results():
    """Visualize and save augmentation results."""
    
    print("Generating augmentation visualization...")
    
    # Create augmentation folder
    augmentation_dir = "augmentation"
    os.makedirs(augmentation_dir, exist_ok=True)
    
    # Load YOLO model
    yolo_model = YOLO('models/yolo_finetuned/best.pt')
    
    # Generate dataset
    print("Loading dataset...")
    img_arr, rgb_img_arr, keypoints_img_arr, file_paths = generate_dataset_data(
        "via_proj/via_project_22Aug2025_16h07m06s.json",
        "RGB-images-jpg/",
        yolo_model,
        [1]
    )
    
    print(f"Dataset loaded: {len(img_arr)} samples")
    print(f"RGB images shape: {rgb_img_arr.shape}")
    print(f"Keypoints shape: {keypoints_img_arr.shape}")
    
    # Create datasets with different augmentations
    no_aug_dataset = KeypointDataset(img_arr, rgb_img_arr, keypoints_img_arr, file_paths, transform=None)
    minimal_aug_dataset = KeypointDataset(img_arr, rgb_img_arr, keypoints_img_arr, file_paths, transform=MinimalAugmentation())
    stronger_aug_dataset = KeypointDataset(img_arr, rgb_img_arr, keypoints_img_arr, file_paths, transform=StrongerAugmentation())
    
    # Test individual augmentations
    rotate_flip = RandomRotateFlip(p=1.0)  # Always apply
    color_jitter = ColorJitter(brightness=0.2, contrast=0.2, p=1.0)  # Stronger for visualization
    noise = GaussianNoise(std=5.0, p=1.0)  # Always apply
    cutout = Cutout(p=1.0, size=32)  # Always apply
    
    # Select a few sample images
    sample_indices = [0, 5, 10, 15, 20]  # Sample 5 images
    
    for idx in sample_indices:
        print(f"Processing sample {idx}...")
        
        # Get original sample
        original_sample = no_aug_dataset[idx]
        original_img = original_sample['image'].numpy().transpose(1, 2, 0)
        original_keypoints = original_sample['keypoints'].numpy()
        
        # Create figure for this sample
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Augmentation Results - Sample {idx}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(original_img.astype(np.uint8))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Original keypoints heatmap
        axes[1, 0].imshow(original_keypoints, cmap='hot')
        axes[1, 0].set_title('Original Keypoints')
        axes[1, 0].axis('off')
        
        # Test fixed threshold function on original
        peaks_original = fixed_threshold_locations(original_keypoints, threshold=0.1)
        if peaks_original:
            for peak in peaks_original:
                axes[1, 0].plot(peak[1], peak[0], 'rx', markersize=10, markeredgewidth=2)
        
        # Minimal augmentation
        try:
            minimal_sample = minimal_aug_dataset[idx]
            minimal_img = minimal_sample['image'].numpy().transpose(1, 2, 0)
            minimal_keypoints = minimal_sample['keypoints'].numpy()
            
            axes[0, 1].imshow(minimal_img.astype(np.uint8))
            axes[0, 1].set_title('Minimal Augmentation')
            axes[0, 1].axis('off')
            
            axes[1, 1].imshow(minimal_keypoints, cmap='hot')
            axes[1, 1].set_title('Minimal Aug Keypoints')
            axes[1, 1].axis('off')
            
            peaks_minimal = fixed_threshold_locations(minimal_keypoints, threshold=0.1)
            if peaks_minimal:
                for peak in peaks_minimal:
                    axes[1, 1].plot(peak[1], peak[0], 'rx', markersize=10, markeredgewidth=2)
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[0, 1].set_title('Minimal Augmentation (Error)')
            axes[1, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[1, 1].set_title('Minimal Aug Keypoints (Error)')
        
        # Stronger augmentation
        try:
            stronger_sample = stronger_aug_dataset[idx]
            stronger_img = stronger_sample['image'].numpy().transpose(1, 2, 0)
            stronger_keypoints = stronger_sample['keypoints'].numpy()
            
            axes[0, 2].imshow(stronger_img.astype(np.uint8))
            axes[0, 2].set_title('Stronger Augmentation')
            axes[0, 2].axis('off')
            
            axes[1, 2].imshow(stronger_keypoints, cmap='hot')
            axes[1, 2].set_title('Stronger Aug Keypoints')
            axes[1, 2].axis('off')
            
            peaks_stronger = fixed_threshold_locations(stronger_keypoints, threshold=0.1)
            if peaks_stronger:
                for peak in peaks_stronger:
                    axes[1, 2].plot(peak[1], peak[0], 'rx', markersize=10, markeredgewidth=2)
        except Exception as e:
            axes[0, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[0, 2].set_title('Stronger Augmentation (Error)')
            axes[1, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[1, 2].set_title('Stronger Aug Keypoints (Error)')
        
        # Individual augmentation: Cutout
        try:
            cutout_sample = cutout(original_sample.copy())
            cutout_img = cutout_sample['image'].numpy().transpose(1, 2, 0)
            cutout_keypoints = cutout_sample['keypoints'].numpy()
            
            axes[0, 3].imshow(cutout_img.astype(np.uint8))
            axes[0, 3].set_title('Cutout Augmentation')
            axes[0, 3].axis('off')
            
            axes[1, 3].imshow(cutout_keypoints, cmap='hot')
            axes[1, 3].set_title('Cutout Keypoints')
            axes[1, 3].axis('off')
            
            peaks_cutout = fixed_threshold_locations(cutout_keypoints, threshold=0.1)
            if peaks_cutout:
                for peak in peaks_cutout:
                    axes[1, 3].plot(peak[1], peak[0], 'rx', markersize=10, markeredgewidth=2)
        except Exception as e:
            axes[0, 3].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[0, 3].set_title('Cutout Augmentation (Error)')
            axes[1, 3].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[1, 3].set_title('Cutout Keypoints (Error)')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(augmentation_dir, f'augmentation_sample_{idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save individual images for easier viewing
        individual_dir = os.path.join(augmentation_dir, f'sample_{idx}')
        os.makedirs(individual_dir, exist_ok=True)
        
        # Save original
        plt.imsave(os.path.join(individual_dir, 'original.png'), original_img.astype(np.uint8))
        plt.imsave(os.path.join(individual_dir, 'original_keypoints.png'), original_keypoints, cmap='hot')
        
        # Save minimal augmentation
        try:
            plt.imsave(os.path.join(individual_dir, 'minimal_aug.png'), minimal_img.astype(np.uint8))
            plt.imsave(os.path.join(individual_dir, 'minimal_keypoints.png'), minimal_keypoints, cmap='hot')
        except:
            pass
        
        # Save stronger augmentation
        try:
            plt.imsave(os.path.join(individual_dir, 'stronger_aug.png'), stronger_img.astype(np.uint8))
            plt.imsave(os.path.join(individual_dir, 'stronger_keypoints.png'), stronger_keypoints, cmap='hot')
        except:
            pass
        
        # Save cutout
        try:
            plt.imsave(os.path.join(individual_dir, 'cutout.png'), cutout_img.astype(np.uint8))
            plt.imsave(os.path.join(individual_dir, 'cutout_keypoints.png'), cutout_keypoints, cmap='hot')
        except:
            pass
    
    print(f"Augmentation visualizations saved to '{augmentation_dir}' folder")
    
    # Test threshold function on a few samples
    print("\nTesting fixed threshold function...")
    for idx in [0, 5, 10]:
        sample = no_aug_dataset[idx]
        keypoints = sample['keypoints'].numpy()
        
        print(f"\nSample {idx}:")
        print(f"  Keypoints shape: {keypoints.shape}")
        print(f"  Keypoints min/max: {keypoints.min():.4f}/{keypoints.max():.4f}")
        print(f"  Keypoints mean: {keypoints.mean():.4f}")
        
        peaks = fixed_threshold_locations(keypoints, threshold=0.1)
        print(f"  Detected peaks: {peaks}")
        
        # Test different thresholds
        for threshold in [0.05, 0.1, 0.2, 0.5]:
            peaks_thresh = fixed_threshold_locations(keypoints, threshold=threshold)
            print(f"  Threshold {threshold}: {len(peaks_thresh)} peaks")

if __name__ == "__main__":
    visualize_augmentation_results()
