# Bed Sheet Folding Robot - Keypoint Detection

A comprehensive keypoint detection system for bed sheet folding robots using deep learning and computer vision techniques.

## 🚀 Features

- **Hybrid Keypoint Detection Model**: YOLO + Vision Transformer architecture
- **Two-Stage Training Pipeline**: Pretraining + Post-training optimization
- **Optimized Training Pipeline**: With `torch.compile()`, early stopping, and mixup augmentation
- **Real-time Inference**: Optimized for real-time bed sheet keypoint detection
- **Comprehensive Data Processing**: YOLO segmentation + keypoint annotation pipeline

## 📁 Project Structure

```
bedSheetFoldingRobot/
├── src/
│   ├── models/                 # Model architectures
│   │   ├── hybrid_keypoint_net.py
│   │   └── efficient_keypoint_net.py
│   ├── utils/                  # Utility functions
│   │   ├── model_utils.py      # YOLO backbone and model utilities
│   │   └── tensorrt_utils.py   # TensorRT conversion utilities (future)
│   └── training/               # Training pipeline
├── shared/                     # Shared functions and utilities
├── models/                     # Trained models and YOLO weights
├── data/                       # Dataset and annotations
├── results/                    # Training results and visualizations
├── keypoint_detection_model_training.py  # Stage 1: Pretraining script
├── post_keypoint_detection_model_training.py  # Stage 2: Post-training script
├── convert_to_tensorrt.py      # TensorRT conversion script (future)
└── test_tensorrt_inference.py  # Performance benchmarking (future)
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd bedSheetFoldingRobot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Stage 1: Pretraining

First, you need to pretrain the model using the original training script:

```bash
# Stage 1: Pretrain the model
python keypoint_detection_model_training.py
```

This will:
- Train the model from scratch on your dataset
- Save the pretrained model to `models/keypoint_model_vit.pth`
- Establish the baseline performance

### Stage 2: Post-Training Optimization

After pretraining, use the post-training script for optimization:

```bash
# Stage 2: Post-training with optimizations
python post_keypoint_detection_model_training.py config_quantization_fixed.json
```

**Configuration Options:**
- `num_epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size for training (default: 16)
- `learning_rate`: Learning rate (default: 0.001)
- `early_stopping_patience`: Early stopping patience (default: 10)
- `use_mixup`: Enable mixup augmentation (default: true)

## 🏗️ Model Architecture

### Hybrid Keypoint Network
- **Backbone**: YOLO11L-pose (first 12 layers)
- **Head**: Vision Transformer for keypoint detection
- **Output**: Heatmap-based keypoint predictions
- **Input**: 128x128 RGB images
- **Parameters**: ~100M parameters

### Key Features
- **torch.compile()**: Optimized training with PyTorch 2.0 compilation
- **Early Stopping**: Automatic training termination on validation loss plateau
- **Mixup Augmentation**: Improved generalization with mixup data augmentation
- **Best Model Saving**: Automatically saves the best model based on validation loss

## 📊 Performance

### Training Optimizations
- **torch.compile()**: ~20-30% faster training
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Clipping**: Stable training with gradient clipping
- **Learning Rate Scheduling**: Adaptive learning rate scheduling

## 🔧 Configuration

### Training Configuration (`config_quantization_fixed.json`)
```json
{
    "model_name": "HybridKeypointNet",
    "model_save_path": "models/keypoint_model_vit_post",
    "pretrained_model_path": "models/keypoint_model_vit.pth",
    "yolo_model_path": "models/yolo_finetuned/best.pt",
    "keypoints_data_src": "via_proj/via_project_22Aug2025_16h07m06s.json",
    "image_path": "RGB-images-jpg/",
    "allowed_classes": [1],
    "batch_size": 16,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "use_augmentation": true,
    "use_mixup": true,
    "early_stopping_patience": 10
}
```

## 📈 Training Pipeline

### Stage 1: Pretraining
1. **Data Loading**: Load images and keypoint annotations
2. **YOLO Segmentation**: Extract bed sheet masks using fine-tuned YOLO
3. **Model Training**: Train from scratch with basic optimizations
4. **Model Saving**: Save pretrained model for post-training

### Stage 2: Post-Training
1. **Load Pretrained Model**: Load from Stage 1 results
2. **Advanced Augmentation**: Apply rotation, flipping, and mixup
3. **Optimized Training**: Train with torch.compile() and early stopping
4. **Evaluation**: Evaluate on test set and visualize results

## 🚀 Deployment

### Production Deployment
1. **Complete Stage 1**: Pretrain the model
2. **Complete Stage 2**: Post-train with optimizations
3. **Deploy Model**: Use the final optimized model for inference

### Real-time Inference
```python
import torch
from src.models import HybridKeypointNet

# Load trained model
model = HybridKeypointNet(...)
model.load_state_dict(torch.load("models/keypoint_model_vit_post.pth"))
model.eval()

# Run inference
with torch.no_grad():
    output = model(input_tensor)
```

## 📝 Usage Examples

### Complete Training Workflow
```bash
# Step 1: Pretraining
python keypoint_detection_model_training.py

# Step 2: Post-training optimization
python post_keypoint_detection_model_training.py config_quantization_fixed.json
```

### Custom Configuration
```python
# Modify config_quantization_fixed.json for your needs
{
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "early_stopping_patience": 20
}
```

## 🔮 Future Improvements

### Planned Features
- **TensorRT Optimization**: 2-5x faster inference with TensorRT conversion
- **Quantization Support**: INT8 quantization for edge deployment
- **Model Export**: ONNX and TorchScript export capabilities
- **Advanced Augmentation**: More sophisticated data augmentation strategies
- **Active Learning**: Uncertainty sampling for efficient training

### TensorRT Integration (Future)
```bash
# Convert to TensorRT for optimized inference
python convert_to_tensorrt.py \
    --model_path models/keypoint_model_vit_post.pth \
    --precision fp16 \
    --test_inference

# Benchmark performance
python test_tensorrt_inference.py \
    --pytorch_model models/keypoint_model_vit_post.pth \
    --tensorrt_model models/keypoint_model_vit_post.trt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO architecture by Ultralytics
- Vision Transformer by Google Research
- PyTorch by Facebook Research
