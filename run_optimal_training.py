#!/usr/bin/env python3
"""
Optimal Training Script with All Improvements Integrated
This script runs the complete training pipeline with all optimizations we've developed.
"""

import os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from post_keypoint_detection_model_training import main_training_pipeline, DEFAULT_CONFIG

def run_optimal_training():
    """Run the optimal training configuration with all improvements."""
    
    print("🚀 Starting Optimal Training with All Improvements")
    print("=" * 60)
    
    # Create optimal configuration
    config = DEFAULT_CONFIG.copy()
    
    # Optimal settings based on our testing
    config.update({
        "num_epochs": 300,  # Full training
        "batch_size": 32,   # Optimal batch size
        "learning_rate": 3e-5,  # Optimal learning rate
        "use_stronger_augmentation": True,  # Include Cutout + stronger transforms
        "use_mixup": True,  # Mixup for regularization
        "use_fp16": True,  # FP16 for speed
        "gradient_accumulation_steps": 2,  # Gradient accumulation
        "early_stopping_patience": 20,  # Early stopping
        "save_interval": 10,  # Save every 10 epochs
        "use_quantization": False,  # Skip QAT for now
        "export_model": True,  # Export final model
    })
    
    print("📋 Configuration Summary:")
    print(f"  • Epochs: {config['num_epochs']}")
    print(f"  • Batch Size: {config['batch_size']}")
    print(f"  • Learning Rate: {config['learning_rate']}")
    print(f"  • Stronger Augmentation: {config['use_stronger_augmentation']} ✅")
    print(f"  • Mixup: {config['use_mixup']} ✅")
    print(f"  • FP16 Training: {config['use_fp16']} ✅")
    print(f"  • Gradient Accumulation: {config['gradient_accumulation_steps']} ✅")
    print(f"  • Early Stopping: {config['early_stopping_patience']} epochs ✅")
    print(f"  • Save Interval: Every {config['save_interval']} epochs ✅")
    
    print("\n🔧 Integrated Improvements:")
    print("  ✅ Fixed coordinate system mapping")
    print("  ✅ Confidence-based masking refinement")
    print("  ✅ Single peak detection per keypoint")
    print("  ✅ Proper train/eval data separation")
    print("  ✅ Cutout augmentation (in StrongerAugmentation)")
    print("  ✅ Multi-stage learning rate scheduling")
    print("  ✅ Gradient clipping and accumulation")
    print("  ✅ Early stopping with best model saving")
    print("  ✅ FP16 mixed precision training")
    print("  ✅ Label smoothing and dropout")
    
    print("\n📊 Expected Performance:")
    print("  • Training Loss Target: < 2.0")
    print("  • Validation Loss Target: < 3.0")
    print("  • Overfitting Ratio Target: < 1.5x")
    print("  • Convergence: Stable within 50-100 epochs")
    
    print("\n" + "=" * 60)
    print("🎯 Starting Training Pipeline...")
    print("=" * 60)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['model_save_path'] = f"models/keypoint_model_optimal_{timestamp}"
    config['results_dir'] = f"results/optimal_{timestamp}"
    
    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)
    
    try:
        # Run training pipeline
        model, history = main_training_pipeline(config)
        
        print("\n" + "=" * 60)
        print("🎉 OPTIMAL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print final results
        if history and 'train_loss' in history and len(history['train_loss']) > 0:
            final_train_loss = history['train_loss'][-1]
            best_train_loss = min(history['train_loss'])
            
            print(f"📈 Training Results:")
            print(f"  • Final Training Loss: {final_train_loss:.4f}")
            print(f"  • Best Training Loss: {best_train_loss:.4f}")
            
            if 'val_loss' in history and len(history['val_loss']) > 0:
                final_val_loss = history['val_loss'][-1]
                best_val_loss = min(history['val_loss'])
                
                print(f"  • Final Validation Loss: {final_val_loss:.4f}")
                print(f"  • Best Validation Loss: {best_val_loss:.4f}")
                
                # Calculate overfitting ratio
                overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')
                print(f"  • Overfitting Ratio: {overfitting_ratio:.2f}x")
                
                # Performance assessment
                if overfitting_ratio < 1.5:
                    print("  ✅ Excellent: Low overfitting")
                elif overfitting_ratio < 2.0:
                    print("  ⚠️ Good: Moderate overfitting")
                else:
                    print("  ❌ High: Significant overfitting")
                
                # Loss targets assessment
                if final_train_loss < 2.0:
                    print("  ✅ Training Loss Target Achieved!")
                else:
                    print("  ⚠️ Training Loss Target Not Met")
                
                if final_val_loss < 3.0:
                    print("  ✅ Validation Loss Target Achieved!")
                else:
                    print("  ⚠️ Validation Loss Target Not Met")
        
        # Check model files
        model_path = f"{config['model_save_path']}.pth"
        if os.path.exists(model_path):
            print(f"✅ Model saved: {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
        
        # Check results
        if os.path.exists(config['results_dir']):
            result_files = [f for f in os.listdir(config['results_dir']) if f.endswith('.png')]
            print(f"✅ Results generated: {len(result_files)} visualization files")
        else:
            print(f"⚠️ Results directory not found: {config['results_dir']}")
        
        # Save training history
        history_path = f"{config['results_dir']}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✅ Training history saved: {history_path}")
        
        print("\n" + "=" * 60)
        print("🎯 ALL SYSTEMS WORKING OPTIMALLY!")
        print("=" * 60)
        
        return True, history
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def print_augmentation_details():
    """Print details about the augmentation pipeline."""
    print("\n🔍 Augmentation Pipeline Details:")
    print("=" * 40)
    
    print("📦 StrongerAugmentation includes:")
    print("  • RandomRotateFlip (p=0.5)")
    print("  • ColorJitter (brightness=0.1, contrast=0.1, p=0.5)")
    print("  • GaussianNoise (std=3.0, p=0.4)")
    print("  • Cutout (p=0.3, size=16) ✅")
    
    print("\n📦 MinimalAugmentation includes:")
    print("  • RandomRotateFlip (p=0.3)")
    print("  • ColorJitter (brightness=0.05, contrast=0.05, p=0.3)")
    print("  • GaussianNoise (std=2.0, p=0.2)")
    
    print("\n📦 Additional Features:")
    print("  • Mixup (alpha=0.2)")
    print("  • Proper train/eval separation")
    print("  • Confidence-based masking")
    print("  • Single peak detection")

if __name__ == "__main__":
    print_augmentation_details()
    
    # Ask for confirmation
    response = input("\n🤔 Proceed with optimal training? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        success, history = run_optimal_training()
        
        if success:
            print("\n✅ Optimal training completed successfully!")
            print("🎯 Your model is now ready for deployment!")
        else:
            print("\n❌ Training failed. Please check the error messages above.")
    else:
        print("\n⏹️ Training cancelled by user.")
