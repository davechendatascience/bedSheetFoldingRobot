# Bed Sheet Folding Robot - Keypoint Detection System

A comprehensive deep learning system for detecting keypoints on bed sheets to enable automated folding robots. This project implements state-of-the-art keypoint detection using hybrid YOLO-ViT architectures with quantization-aware training for efficient deployment.

## 🎯 Project Overview

This system detects keypoints on bed sheets from RGB and depth images, enabling precise robotic manipulation for automated bed sheet folding. The project includes:

- **Hybrid Keypoint Detection Models** (YOLO + ViT)
- **Quantization-Aware Training (QAT)** for efficient deployment
- **Multi-modal Input Support** (RGB + Depth)
- **Comprehensive Training Pipeline**
- **Model Export** (ONNX, GGUF formats)

## 🏗️ Architecture

### Model Structure
```
Input (128x128) → YOLO Backbone → Feature Fusion → ViT Encoder → Decoder → Spatial Softmax → Keypoint Heatmaps
```

### Key Components
- **YOLO Backbone**: Feature extraction from multiple scales
- **MultiScaleFusion**: Combines features from different YOLO layers
- **ViT Encoder**: Vision Transformer for global context
- **SingleHeatmapDecoder**: Upsamples to full resolution
- **Spatial Softmax**: Converts logits to probability heatmaps

## 📊 Model Specifications

### Current Model (HybridKeypointNet)
- **Parameters**: ~103M (includes ViT-B encoder)
- **Input**: 128x128 RGB/Depth images
- **Output**: 128x128 keypoint heatmaps
- **Architecture**: YOLO-L + ViT-B + Decoder

### Quantization Support
- **QAT Training**: Quantization-aware training pipeline
- **Model Export**: ONNX and GGUF formats
- **Reduced Precision**: 8-bit quantization for deployment

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training Data Generation
```bash
cd cloth_data_gen
blender --background --python cloth_dataset_gen.py
```

### Regular Training
```bash
# Train the full model (103M parameters)
python keypoint_detection_model_training.py
```

### Quantized Training
```bash
# Train with quantization-aware training
python depth_keypoint_model_training.py config_quantization_fixed.json

# Or use the runner script
python run_quantized_depth_training.py
```

## 📁 Project Structure

```
bedSheetFoldingRobot/
├── src/                          # Core source code
│   ├── models/                   # Model architectures
│   │   ├── hybrid_keypoint_net.py
│   │   └── __init__.py
│   ├── training/                 # Training pipeline
│   │   ├── training_pipeline.py
│   │   └── __init__.py
│   └── utils/                    # Utilities
│       ├── model_utils.py
│       ├── quantization_utils.py
│       └── __init__.py
├── cloth_data_gen/              # Data generation
├── realsense/                   # Depth camera utilities
├── via_proj/                    # VIA annotation projects
├── models/                      # Trained models
├── results/                     # Training results
├── config_quantization_fixed.json  # QAT configuration
├── depth_keypoint_model_training.py # Main training script
└── run_quantized_depth_training.py  # QAT runner
```

## 🔧 Training Configuration

### Regular Training
- **Learning Rate**: 3e-5
- **Batch Size**: 32
- **Epochs**: 300
- **Optimizer**: AdamW
- **Loss**: KL Divergence with Gaussian blur

### Quantization-Aware Training
- **QAT Epochs**: 150
- **QAT Learning Rate**: 5e-5
- **Backbone**: Unfrozen (full fine-tuning)
- **Adaptive LR Schedule**: Warmup + Plateau + Decay

## 📈 Performance

### Training Metrics
- **Dataset Size**: 2,999 images
- **Train/Test Split**: 80/20
- **Convergence**: ~50-100 epochs
- **Final Loss**: ~2.5-3.0 (regular), ~10.0-12.0 (quantized)

### Model Efficiency
- **Regular Model**: 103M parameters, ~400MB memory
- **Quantized Model**: Same architecture, ~100MB memory
- **Inference Speed**: Real-time capable (30+ FPS)

## 🎛️ Configuration Files

### `config_quantization_fixed.json`
```json
{
  "use_quantization": true,
  "qat_epochs": 150,
  "qat_learning_rate": 5e-5,
  "freeze_backbone": false,
  "export_formats": ["onnx", "gguf"]
}
```

## 🔬 Key Features

### 1. Quantization-Aware Training
- **Automatic QAT**: Converts models to quantized format
- **Adaptive Learning**: Specialized LR schedules for QAT
- **Weight Loading**: Preserves pretrained knowledge
- **Export Pipeline**: ONNX and GGUF formats

### 2. Multi-Modal Support
- **RGB Images**: Standard color images
- **Depth Images**: RealSense depth data
- **Hybrid Processing**: Combined feature extraction

### 3. Advanced Training Features
- **Data Augmentation**: Rotation, flip, color jitter
- **Mixup Training**: Improves generalization
- **FP16 Training**: Accelerated training
- **Gradient Clipping**: Training stability

### 4. Model Export
- **ONNX Export**: Cross-platform deployment
- **GGUF Export**: Optimized for edge devices
- **Quantization**: 8-bit precision for efficiency

## 🧪 Usage Examples

### Basic Training
```python
from depth_keypoint_model_training import main_training_pipeline
from config_quantization_fixed import DEFAULT_CONFIG

# Train with quantization
config = DEFAULT_CONFIG.copy()
config["use_quantization"] = True
model, history = main_training_pipeline(config)
```

### Model Inference
```python
import torch
from src.models import HybridKeypointNet

# Load trained model
model = HybridKeypointNet(backbone, in_channels_list)
model.load_state_dict(torch.load("models/keypoint_model_vit_depth_quantized.pth"))

# Inference
with torch.no_grad():
    heatmaps = model(input_image)
    keypoints = soft_argmax(heatmaps)
```

## 🔍 Model Analysis

### Architecture Comparison
| Model Type | Parameters | Use Case | Memory |
|------------|------------|----------|---------|
| **Current Hybrid** | 103M | Full accuracy | 400MB |
| **Quantized** | 103M | Efficient deployment | 100MB |
| **Recommended** | 5-10M | Balanced approach | 40MB |

### Why Current Model is Large
- **ViT-B Encoder**: 86M parameters (designed for 224x224)
- **YOLO-L Backbone**: 12 layers for feature extraction
- **Double Encoding**: YOLO + ViT (redundant for keypoints)

## 🚧 Known Issues & Solutions

### 1. Model Size
- **Issue**: 103M parameters for keypoint detection
- **Solution**: Use efficient architectures (5-10M parameters)

### 2. Quantization Convergence
- **Issue**: QAT loss higher than regular training
- **Solution**: Expected behavior - quantization noise

### 3. Training Time
- **Issue**: Slow training with large model
- **Solution**: Use smaller models or distributed training

## 🔮 Future Improvements

### Planned Features
1. **Efficient Architectures**: 5-10M parameter models
2. **Distributed Training**: Multi-GPU support
3. **Real-time Inference**: Optimized deployment
4. **Edge Deployment**: Mobile/embedded support

### Architecture Optimizations
1. **Remove ViT**: Use only YOLO backbone
2. **Lightweight Decoder**: Reduce upsampling layers
3. **Feature Pruning**: Remove redundant features
4. **Knowledge Distillation**: Transfer learning

## 📚 Documentation

- **Training Guide**: `quantization_vs_regular_explanation.md`
- **Architecture Details**: `model_architecture_diagram.py`
- **Data Flow**: `data_flow_pipeline.pdf`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLO**: Ultralytics for object detection backbone
- **ViT**: Vision Transformer architecture
- **RealSense**: Intel for depth camera support
- **VIA**: VGG Image Annotator for data labeling

---

**Note**: This project is actively developed. The current 103M parameter model is being optimized for efficiency while maintaining accuracy.
