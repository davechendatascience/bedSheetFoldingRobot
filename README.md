# Bed Sheet Folding Robot - Keypoint Detection System

A comprehensive computer vision system for detecting keypoints on bed sheets to enable robotic folding operations. This project implements state-of-the-art deep learning models for precise keypoint detection using both RGB and depth data.

## 🚀 Features

- **Hybrid Architecture**: Combines YOLO backbone with Vision Transformer encoder for robust feature extraction
- **Multiple Model Variants**: From lightweight to full-scale models for different deployment scenarios
- **Quantization Support**: Post-training quantization (PTQ) and quantization-aware training (QAT) for efficient deployment
- **Advanced Training Pipeline**: Comprehensive training with validation, monitoring, and early stopping
- **Data Augmentation**: Extensive augmentation strategies including MixUp, Cutout, and Albumentations
- **Active Learning**: Uncertainty sampling for improved training efficiency
- **Model Export**: Support for ONNX and GGUF formats for deployment

## 📁 Project Structure

```
bedSheetFoldingRobot/
├── src/
│   ├── models/
│   │   ├── hybrid_keypoint_net.py      # Main hybrid architecture
│   │   ├── efficient_keypoint_net.py   # Lightweight model variants
│   │   └── __init__.py
│   ├── training/
│   │   └── training_utils.py
│   └── utils/
│       ├── quantization_utils.py       # Quantization utilities
│       ├── model_utils.py              # Model utilities
│       └── functions.py                # Shared functions
├── models/                             # Trained models and checkpoints
├── via_proj/                          # VIA annotation files
├── RGB-images-jpg/                    # Training images
├── realsense/                         # Depth data processing
├── results/                           # Training results and visualizations
├── post_keypoint_detection_model_training.py  # Main training pipeline
├── run_optimal_training.py            # Training execution script
├── run_quantized_depth_training.py    # Quantized training script
├── config_quantization_fixed.json     # Quantization configuration
└── requirements.txt                   # Python dependencies
```

## 🏗️ Model Architectures

### HybridKeypointNet (Main Model)
- **Backbone**: YOLOv8 with fine-tuned weights
- **Encoder**: Vision Transformer for global context
- **Decoder**: Multi-scale feature fusion
- **Parameters**: ~103M parameters
- **Use Case**: High-accuracy applications

### Efficient Model Variants
- **EfficientKeypointNet**: Optimized for speed
- **UltraLightKeypointNet**: Minimal footprint
- **MobileKeypointNet**: Mobile deployment
- **EfficientViTKeypointNet**: Lightweight ViT variant

## 🎯 Training Pipeline

### Main Training Script: `post_keypoint_detection_model_training.py`

This script provides a comprehensive training pipeline that integrates:
- **Regular Training**: Full model training with advanced techniques
- **Quantization Training**: Seamless QAT integration
- **Validation & Monitoring**: Comprehensive evaluation
- **Model Export**: Deployment-ready models

#### Key Features:
- **torch.compile()**: JIT compilation for performance
- **Mixed Precision Training**: FP16 training with automatic mixed precision
- **Active Learning**: Uncertainty sampling for efficient training
- **Advanced Augmentation**: MixUp, Cutout, ColorJitter, GaussianNoise
- **Learning Rate Scheduling**: Warmup and cosine annealing
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Stable training

### Training Flow

1. **Regular Training**:
   ```bash
   python post_keypoint_detection_model_training.py
   ```
   - Loads pretrained model (if available)
   - Trains for 300 epochs (configurable)
   - Saves to `models/keypoint_model_vit_post.pth`

2. **Quantization Training** (if enabled):
   - Automatically continues after regular training
   - Loads from regular training result
   - Trains for 50 epochs with quantization simulation
   - Saves QAT model and final quantized model

## ⚙️ Configuration

### Default Configuration
```python
DEFAULT_CONFIG = {
    "seed": 42,
    "yolo_model_path": "models/yolo_finetuned/best.pt",
    "keypoints_data_src": "via_proj/via_project_22Aug2025_16h07m06s.json",
    "image_path": "RGB-images-jpg/",
    "batch_size": 32,
    "learning_rate": 3e-5,
    "num_epochs": 300,
    "model_save_path": "models/keypoint_model_vit_post.pth",
    "use_quantization": False,
    "qat_epochs": 50,
    "qat_learning_rate": 5e-5,
    # ... additional parameters
}
```

### Quantization Configuration
```json
{
    "use_quantization": true,
    "qat_epochs": 50,
    "qat_learning_rate": 5e-5,
    "export_model": true,
    "export_formats": ["onnx", "gguf"]
}
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate your Python environment
source ~/pytorch_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Regular Training
```bash
python post_keypoint_detection_model_training.py
```

### 3. Training with Quantization
```bash
# Edit config to enable quantization
python post_keypoint_detection_model_training.py
```

### 4. Using Configuration Files
```bash
# For quantization training
python run_quantized_depth_training.py

# For optimal training
python run_optimal_training.py
```

## 📊 Model Performance

### Training Features
- **Loss Function**: KL divergence with spatial softmax
- **Optimization**: AdamW with weight decay
- **Scheduling**: Warmup + cosine annealing
- **Regularization**: Dropout, label smoothing
- **Monitoring**: Training curves, validation metrics

### Quantization Benefits
- **Model Size**: ~75% reduction
- **Inference Speed**: 2-4x faster
- **Memory Usage**: Significantly reduced
- **Accuracy**: Minimal degradation (<1%)

## 🔧 Advanced Features

### Data Augmentation Pipeline
- **RandomRotateFlip**: Geometric transformations
- **ColorJitter**: Color space augmentation
- **GaussianNoise**: Noise injection
- **Cutout**: Occlusion simulation
- **MixUp**: Sample mixing for regularization
- **StrongerAugmentation**: Albumentations integration

### Active Learning
- **Uncertainty Sampling**: Entropy-based sample selection
- **Dynamic Batch Selection**: Focuses on challenging samples
- **Efficient Training**: Reduces required training samples

### Model Export
- **ONNX Format**: Cross-platform deployment
- **GGUF Format**: Optimized for inference
- **Quantized Models**: Ready for edge deployment

## 📈 Training Monitoring

The training pipeline provides comprehensive monitoring:
- **Real-time Loss Tracking**: Per-epoch loss visualization
- **Validation Metrics**: Regular evaluation on test set
- **Model Checkpoints**: Automatic saving at intervals
- **Performance Comparison**: Regular vs quantized model metrics
- **Visualization**: Keypoint detection results

## 🛠️ Development

### Adding New Models
1. Create model class in `src/models/`
2. Add to `src/models/__init__.py`
3. Update training pipeline if needed

### Custom Training
1. Modify `DEFAULT_CONFIG` in training script
2. Add custom augmentation in augmentation classes
3. Implement custom loss functions if needed

### Quantization Customization
1. Adjust QAT parameters in configuration
2. Modify quantization utilities in `src/utils/quantization_utils.py`
3. Add custom quantization schemes

## 📝 Recent Updates

### Latest Improvements
- ✅ **Seamless Training Pipeline**: Regular + QAT in single script
- ✅ **Model Preservation**: No overwriting of pretrained models
- ✅ **Advanced Augmentation**: Comprehensive data augmentation
- ✅ **Performance Optimization**: torch.compile and mixed precision
- ✅ **Quantization Support**: Full QAT pipeline with 50 epochs
- ✅ **Active Learning**: Uncertainty sampling for efficient training
- ✅ **Model Export**: ONNX and GGUF support

### Architecture Changes
- **HybridKeypointNet**: Main model with YOLO + ViT
- **Efficient Variants**: Lightweight alternatives
- **Quantization Pipeline**: Post-training quantization support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 for backbone architecture
- Vision Transformer for global context modeling
- PyTorch for the deep learning framework
- VIA for annotation tools
- RealSense for depth data collection

---

**Note**: This project is actively maintained and updated with the latest deep learning techniques for optimal keypoint detection performance.
