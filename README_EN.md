# Enterprise-Level Universal Vision Classification Framework

A high-performance image classification training framework based on PyTorch, designed for enterprises and research institutions, covering the complete workflow from data processing, model training to deployment.

## 🚀 Key Features

- **🔧 Universal Design**: Support for any number of classification tasks (2 to 1000+ classes), one-click dataset replacement
- **⚡ Cross-Platform Optimization**: Intelligent adaptation to CUDA/MPS/CPU, 3-5x training speed improvement  
- **🏗️ Enterprise Architecture**: Highly configurable, modular design, production-ready
- **📊 Multi-Model Support**: 7 classic architectures, flexible selection of the most suitable model
- **🎯 Complete Deployment**: One-stop solution from training to GUI applications

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Algorithm Selection Guide](#algorithm-selection-guide)
- [System Adaptation](#system-adaptation)
- [GUI Application Deployment](#gui-application-deployment)
- [Configuration Parameters](#configuration-parameters)
- [FAQ](#faq)

## 🚀 Quick Start

### 1. Environment Installation

```bash
# Clone the project
git clone https://github.com/your-repo/universal-vision-classifier.git
cd universal-vision-classifier

# Create virtual environment
conda create -n vision_classifier python=3.9
conda activate vision_classifier

# Install dependencies (choose based on your system)
# NVIDIA GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (MPS)
pip install torch torchvision torchaudio

# CPU Only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your data according to the following structure:

```
your_dataset/
├── TRAIN/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image3.jpg
│   │   └── ...
│   └── ...
└── TEST/
    ├── class1/
    ├── class2/
    └── ...
```

### 3. One-Click Training

```bash
# Basic training (using default configuration)
python train_garbage.py --dataset_path your_dataset

# Complete training example
python train_garbage.py \
    --dataset_path your_dataset \
    --backbone resnet50 \
    --batch_size 64 \
    --num_epochs 50 \
    --optimizer adamw \
    --use_mixup \
    --balance_samples
```

## 🛠️ Environment Setup

### System Requirements

**Hardware Requirements**:
- CPU: Intel i5/AMD Ryzen 5 or higher
- Memory: 8GB or more (16GB recommended)
- GPU: NVIDIA GPU (recommended) or Apple Silicon chip
- Storage: 10GB available space

**Software Requirements**:
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.7+ (for NVIDIA GPU)

### Detailed Installation Steps

#### 1. NVIDIA GPU Environment

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 2. Apple Silicon Environment

```bash
# Install PyTorch (MPS support)
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### 3. CPU Environment

```bash
# Install CPU version PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Dependency Installation

```bash
# Core dependencies
pip install opencv-python pillow matplotlib numpy
pip install scikit-learn tqdm seaborn pandas
pip install pyserial

# Optional dependencies (for monitoring and visualization)
pip install tensorboard wandb
pip install jupyter notebook
```

## 📁 Data Preparation

### Dataset Format Requirements

The framework uses standard ImageFolder format, supporting the following image formats:
- JPG/JPEG
- PNG
- BMP
- TIFF

### Dataset Example

Using medical image classification as an example:

```
medical_dataset/
├── TRAIN/
│   ├── normal/          # Normal
│   │   ├── normal_001.jpg
│   │   ├── normal_002.jpg
│   │   └── ...
│   ├── pneumonia/       # Pneumonia
│   │   ├── pneumonia_001.jpg
│   │   └── ...
│   └── covid/           # COVID-19
│       ├── covid_001.jpg
│       └── ...
└── TEST/
    ├── normal/
    ├── pneumonia/
    └── covid/
```

### Data Preprocessing Recommendations

1. **Image Size**: Recommend uniform 224x224 or 299x299
2. **Data Balance**: Use `--balance_samples` to handle imbalanced data
3. **Data Augmentation**: Framework automatically applies appropriate data augmentation
4. **Quality Check**: Ensure images are clear and correctly labeled

## 🎯 Model Training

### Basic Training Commands

```bash
# Simplest training command
python train_garbage.py --dataset_path your_dataset

# Specify model architecture
python train_garbage.py \
    --dataset_path your_dataset \
    --backbone efficientnet_b0

# Adjust training parameters
python train_garbage.py \
    --dataset_path your_dataset \
    --backbone resnet50 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001
```

### Advanced Training Configuration

```bash
# Complete configuration example
python train_garbage.py \
    --dataset_path your_dataset \
    --backbone efficientnet_b3 \
    --batch_size 64 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --weight_decay 1e-4 \
    --optimizer adamw \
    --loss_fn label_smoothing \
    --smoothing 0.1 \
    --lr_scheduler cosine \
    --use_mixup \
    --use_cutmix \
    --mixup_alpha 1.0 \
    --balance_samples \
    --use_amp \
    --early_stopping 10 \
    --gradient_accumulation 2
```

### Training Output

After training completion, the following files will be generated:
- `best_model_*.pth`: Best model weights
- `final_model_*.pth`: Final model weights
- `training_results_*.png`: Training curve plots
- Training logs and performance metrics

## 🧠 Algorithm Selection Guide

### Model Architecture Selection

| Model | Parameters | Inference Speed | Accuracy | Use Case |
|-------|------------|-----------------|----------|----------|
| **ResNet18** | 11M | 🚀🚀🚀 | ⭐⭐⭐ | Fast prototyping, real-time applications |
| **ResNet50** | 25M | 🚀🚀 | ⭐⭐⭐⭐ | Balance performance and speed |
| **EfficientNet-B0** | 5M | 🚀🚀🚀 | ⭐⭐⭐⭐ | Mobile deployment |
| **EfficientNet-B3** | 12M | 🚀🚀 | ⭐⭐⭐⭐⭐ | High accuracy requirements |
| **MobileNet-V2** | 3M | 🚀🚀🚀 | ⭐⭐⭐ | Resource-constrained environments |

### Selection Recommendations

#### 📱 Mobile/Edge Devices
```bash
python train_garbage.py \
    --backbone mobilenet_v2 \
    --img_size 224 \
    --batch_size 32
```

#### 🖥️ Server/High Accuracy
```bash
python train_garbage.py \
    --backbone efficientnet_b3 \
    --img_size 300 \
    --batch_size 16 \
    --use_amp
```

#### ⚡ Real-time Applications
```bash
python train_garbage.py \
    --backbone resnet18 \
    --img_size 224 \
    --batch_size 64
```

### Optimizer Selection

| Optimizer | Characteristics | Use Case |
|-----------|-----------------|----------|
| **Adam** | Fast convergence, adaptive learning rate | Default choice for most cases |
| **AdamW** | Improved weight decay | Large models, prevent overfitting |
| **SGD** | Stable, good generalization | Large datasets, long training |

### Loss Function Selection

```bash
# Standard classification
--loss_fn cross_entropy

# Class imbalance
--loss_fn focal_loss --focal_gamma 2.0

# Improve generalization
--loss_fn label_smoothing --smoothing 0.1
```

## 🖥️ System Adaptation

### Automatic Hardware Detection

The framework automatically detects and selects optimal hardware:

```python
# Device priority: CUDA > MPS > CPU
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
```

### NVIDIA GPU Optimization

```bash
# Enable mixed precision training
python train_garbage.py \
    --dataset_path your_dataset \
    --use_amp \
    --batch_size 64

# Multi-GPU training (coming soon)
export CUDA_VISIBLE_DEVICES=0,1
python train_garbage.py --dataset_path your_dataset
```

### Apple Silicon Optimization

```bash
# MPS accelerated training
python train_garbage.py \
    --dataset_path your_dataset \
    --device mps \
    --batch_size 32
```

Note: MPS currently does not support mixed precision training, the framework will automatically disable it.

### CPU Optimization

```bash
# CPU multi-threading optimization
python train_garbage.py \
    --dataset_path your_dataset \
    --device cpu \
    --num_workers 8 \
    --batch_size 16
```

### Memory Optimization

```bash
# Large dataset memory optimization
python train_garbage.py \
    --dataset_path your_dataset \
    --batch_size 16 \
    --gradient_accumulation 4 \
    --num_workers 2
```

## 🖼️ GUI Application Deployment

### Launch GUI Application

```bash
# Start the complete GUI application
python trash_classification_gui_complete.py
```

### GUI Features

- **📁 File Upload**: Support for multiple image formats
- **📹 Real-time Camera**: Real-time classification prediction
- **📸 Photo Mode**: Single image analysis
- **🎲 Random Testing**: Random selection from training set
- **📊 Result Visualization**: Confidence distribution charts
- **🔧 Hardware Integration**: Serial communication control

### Configure GUI Application

Modify configuration in `trash_classification_gui_complete.py`:

```python
# Modify dataset path
self.dataset_path = "your_dataset"

# Modify class names
self.class_names = ['Class1', 'Class2', 'Class3', 'Class4']

# Modify class colors
self.class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
```

## ⚙️ Configuration Parameters

### Model Parameters

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `--backbone` | resnet18 | Model architecture | resnet18/34/50/101, efficientnet_b0/b3, mobilenet_v2 |
| `--batch_size` | 32 | Batch size | 8-128, adjust based on GPU memory |
| `--num_epochs` | 30 | Training epochs | 10-200 |
| `--img_size` | 224 | Input image size | 224, 299, 384 |

### Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | 0.001 | Learning rate |
| `--weight_decay` | 1e-4 | Weight decay |
| `--optimizer` | adam | Optimizer selection |
| `--lr_scheduler` | reduce_on_plateau | Learning rate scheduler |

### Data Augmentation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_mixup` | False | Enable Mixup augmentation |
| `--use_cutmix` | False | Enable CutMix augmentation |
| `--mixup_alpha` | 1.0 | Mixup strength parameter |
| `--balance_samples` | False | Class balanced sampling |

### Performance Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_amp` | False | Mixed precision training |
| `--num_workers` | 4 | Data loading processes |
| `--gradient_accumulation` | 1 | Gradient accumulation steps |
| `--early_stopping` | 0 | Early stopping patience (0 to disable) |

## ❓ FAQ

### Q1: What to do when running out of GPU memory during training?

**Solutions**:
```bash
# Reduce batch size
--batch_size 16

# Use gradient accumulation
--gradient_accumulation 4

# Enable mixed precision training
--use_amp
```

### Q2: How to handle class imbalance?

**Solutions**:
```bash
# Use class balanced sampling
--balance_samples

# Use Focal Loss
--loss_fn focal_loss --focal_gamma 2.0

# Use data augmentation
--use_mixup --use_cutmix
```

### Q3: What to do when training is too slow?

**Solutions**:
```bash
# Increase data loading processes
--num_workers 8

# Use smaller model
--backbone mobilenet_v2

# Enable mixed precision training (NVIDIA GPU)
--use_amp
```

### Q4: How to choose appropriate learning rate?

**Recommendations**:
- Small datasets: 0.001-0.01
- Large datasets: 0.0001-0.001
- Use learning rate scheduling: `--lr_scheduler cosine`

### Q5: What to do about model overfitting?

**Solutions**:
```bash
# Increase regularization
--weight_decay 1e-3

# Use label smoothing
--loss_fn label_smoothing --smoothing 0.1

# Enable data augmentation
--use_mixup --use_cutmix

# Early stopping
--early_stopping 10
```

## 📞 Technical Support

- **GitHub Issues**: [Project Issues Page]
- **Documentation**: [Online Documentation Link]
- **Community Discussion**: [Discussion Forum Link]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 🌟 Acknowledgments

Thanks to all developers and researchers who contributed to this project.

---

**Making AI Technology Accessible to Every Developer!** 🚀
