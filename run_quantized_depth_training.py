#!/usr/bin/env python3
"""
Quantized Depth Model Training Runner

This script runs the depth keypoint model training with quantization enabled.
It uses the existing post_keypoint_detection_model_training.py with quantization configurations.
"""

import sys
import os
import json
from post_keypoint_detection_model_training import main_training_pipeline, DEFAULT_CONFIG

def create_quantized_config():
    """Create configuration for quantized training using existing config file."""
    config = DEFAULT_CONFIG.copy()
    
    # Load quantization configuration from existing file
    config_file = "config_quantization_fixed.json"  # Use fixed config for better convergence
    if os.path.exists(config_file):
        print(f"Loading quantization configuration from {config_file}")
        with open(config_file, 'r') as f:
            quant_config = json.load(f)
        config.update(quant_config)
    else:
        print(f"Warning: {config_file} not found, using default quantization settings")
        # Enable quantization with default settings
        config["use_quantization"] = True
        config["qat_epochs"] = 50
        config["freeze_backbone"] = True
        config["qat_learning_rate"] = 1e-5
        config["pretrained_path"] = "models/keypoint_model_vit.pth"
        config["export_model"] = True
        config["export_name"] = "keypoint_model_quantized"
        config["export_formats"] = ["onnx"]
    
    return config

def main():
    """Main function to run quantized training."""
    print("Starting Quantized Depth Model Training...")
    print("=" * 50)
    
    # Create quantized configuration
    config = create_quantized_config()
    
    # Print configuration
    print("Configuration:")
    print(f"  Use Quantization: {config['use_quantization']}")
    print(f"  QAT Epochs: {config['qat_epochs']}")
    print(f"  Pretrained Path: {config['pretrained_path']}")
    print(f"  Model Save Path: {config['model_save_path']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  QAT Learning Rate: {config['qat_learning_rate']}")
    print(f"  Freeze Backbone: {config['freeze_backbone']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Export Formats: {config['export_formats']}")
    print(f"  Use FP16: {config.get('use_fp16', False)}")
    print("=" * 50)
    
    # Check if pretrained model exists
    if not os.path.exists(config["pretrained_path"]):
        print(f"Warning: Pretrained model not found at {config['pretrained_path']}")
        print("You need to train a regular model first before running quantization.")
        print("Please run the regular training first or update the pretrained_path.")
        return
    
    # Run training pipeline
    try:
        model, history = main_training_pipeline(config)
        
        print("\n" + "=" * 50)
        print("Quantized Training Completed Successfully!")
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"Quantized model saved to: {config['model_save_path']}_quantized.pth")
        
        if config["export_model"]:
            print(f"Model exported to: models/{config['export_name']}.onnx")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
