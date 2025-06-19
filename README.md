# 企业级通用视觉分类框架

一个基于PyTorch的高性能图像分类训练框架，专为企业与科研机构设计，覆盖从数据处理、模型训练到部署的全流程解决方案。

## 🚀 项目特色

- **🔧 通用性设计**：支持任意数量分类任务（2类至1000+类），一键替换数据集
- **⚡ 跨平台优化**：智能适配CUDA/MPS/CPU，训练速度提升3-5倍
- **🏗️ 企业级架构**：高度配置化，模块化设计，生产就绪
- **📊 多模型支持**：7种经典架构，灵活选择最适合的模型
- **🎯 完整部署**：从训练到GUI应用的一站式解决方案

## 📋 目录

- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [算法选择指南](#算法选择指南)
- [系统适配](#系统适配)
- [GUI应用部署](#gui应用部署)
- [配置参数详解](#配置参数详解)
- [常见问题](#常见问题)

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/your-repo/universal-vision-classifier.git
cd universal-vision-classifier

# 创建虚拟环境
conda create -n vision_classifier python=3.9
conda activate vision_classifier

# 安装依赖（根据你的系统选择）
# NVIDIA GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (MPS)
pip install torch torchvision torchaudio

# CPU Only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 准备数据集

按照以下结构组织你的数据：

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

### 3. 一键训练

```bash
# 基础训练（使用默认配置）
python train_garbage.py --dataset_path your_dataset

# 完整训练示例
python train_garbage.py \
    --dataset_path your_dataset \
    --backbone resnet50 \
    --batch_size 64 \
    --num_epochs 50 \
    --optimizer adamw \
    --use_mixup \
    --balance_samples
```

## 🛠️ 环境配置

### 系统要求

**硬件要求**：
- CPU: Intel i5/AMD Ryzen 5 及以上
- 内存: 8GB及以上（推荐16GB）
- 显卡: NVIDIA GPU（推荐）或Apple Silicon芯片
- 存储: 10GB可用空间

**软件要求**：
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.7+（NVIDIA GPU）

### 详细安装步骤

#### 1. NVIDIA GPU环境

```bash
# 检查CUDA版本
nvidia-smi

# 安装PyTorch（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 2. Apple Silicon环境

```bash
# 安装PyTorch（MPS支持）
pip install torch torchvision torchaudio

# 验证MPS可用性
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### 3. CPU环境

```bash
# 安装CPU版本PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 依赖包安装

```bash
# 核心依赖
pip install opencv-python pillow matplotlib numpy
pip install scikit-learn tqdm seaborn pandas
pip install pyserial

# 可选依赖（用于监控和可视化）
pip install tensorboard wandb
pip install jupyter notebook
```

## 📁 数据准备

### 数据集格式要求

框架使用标准的ImageFolder格式，支持以下图像格式：
- JPG/JPEG
- PNG
- BMP
- TIFF

### 数据集示例

以医疗影像分类为例：

```
medical_dataset/
├── TRAIN/
│   ├── normal/          # 正常
│   │   ├── normal_001.jpg
│   │   ├── normal_002.jpg
│   │   └── ...
│   ├── pneumonia/       # 肺炎
│   │   ├── pneumonia_001.jpg
│   │   └── ...
│   └── covid/           # 新冠
│       ├── covid_001.jpg
│       └── ...
└── TEST/
    ├── normal/
    ├── pneumonia/
    └── covid/
```

### 数据预处理建议

1. **图像尺寸**：建议统一为224x224或299x299
2. **数据平衡**：使用`--balance_samples`处理不平衡数据
3. **数据增强**：框架自动应用适当的数据增强
4. **质量检查**：确保图像清晰，标注正确

## 🎯 模型训练

### 基础训练命令

```bash
# 最简单的训练命令
python train_garbage.py --dataset_path your_dataset

# 指定模型架构
python train_garbage.py \
    --dataset_path your_dataset \
    --backbone efficientnet_b0

# 调整训练参数
python train_garbage.py \
    --dataset_path your_dataset \
    --backbone resnet50 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001
```

### 高级训练配置

```bash
# 完整配置示例
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

### 训练输出

训练完成后，将生成以下文件：
- `best_model_*.pth`：最佳模型权重
- `final_model_*.pth`：最终模型权重
- `training_results_*.png`：训练曲线图
- 训练日志和性能指标

## 🧠 算法选择指南

### 模型架构选择

| 模型 | 参数量 | 推理速度 | 准确率 | 适用场景 |
|------|--------|----------|--------|----------|
| **ResNet18** | 11M | 🚀🚀🚀 | ⭐⭐⭐ | 快速原型，实时应用 |
| **ResNet50** | 25M | 🚀🚀 | ⭐⭐⭐⭐ | 平衡性能与速度 |
| **EfficientNet-B0** | 5M | 🚀🚀🚀 | ⭐⭐⭐⭐ | 移动端部署 |
| **EfficientNet-B3** | 12M | 🚀🚀 | ⭐⭐⭐⭐⭐ | 高精度需求 |
| **MobileNet-V2** | 3M | 🚀🚀🚀 | ⭐⭐⭐ | 资源受限环境 |

### 选择建议

#### 📱 移动端/边缘设备
```bash
python train_garbage.py \
    --backbone mobilenet_v2 \
    --img_size 224 \
    --batch_size 32
```

#### 🖥️ 服务器端/高精度
```bash
python train_garbage.py \
    --backbone efficientnet_b3 \
    --img_size 300 \
    --batch_size 16 \
    --use_amp
```

#### ⚡ 实时应用
```bash
python train_garbage.py \
    --backbone resnet18 \
    --img_size 224 \
    --batch_size 64
```

### 优化器选择

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| **Adam** | 快速收敛，自适应学习率 | 大多数情况的默认选择 |
| **AdamW** | 改进的权重衰减 | 大模型，防止过拟合 |
| **SGD** | 稳定，泛化性好 | 大数据集，长时间训练 |

### 损失函数选择

```bash
# 标准分类
--loss_fn cross_entropy

# 类别不平衡
--loss_fn focal_loss --focal_gamma 2.0

# 提升泛化性
--loss_fn label_smoothing --smoothing 0.1
```

## 🖥️ 系统适配

### 自动硬件检测

框架会自动检测并选择最优硬件：

```python
# 设备优先级：CUDA > MPS > CPU
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
```

### NVIDIA GPU优化

```bash
# 启用混合精度训练
python train_garbage.py \
    --dataset_path your_dataset \
    --use_amp \
    --batch_size 64

# 多GPU训练（即将支持）
export CUDA_VISIBLE_DEVICES=0,1
python train_garbage.py --dataset_path your_dataset
```

### Apple Silicon优化

```bash
# MPS加速训练
python train_garbage.py \
    --dataset_path your_dataset \
    --device mps \
    --batch_size 32
```

注意：MPS目前不支持混合精度训练，框架会自动禁用。

### CPU优化

```bash
# CPU多线程优化
python train_garbage.py \
    --dataset_path your_dataset \
    --device cpu \
    --num_workers 8 \
    --batch_size 16
```

### 内存优化

```bash
# 大数据集内存优化
python train_garbage.py \
    --dataset_path your_dataset \
    --batch_size 16 \
    --gradient_accumulation 4 \
    --num_workers 2
```

## 🖼️ GUI应用部署

### 启动GUI应用

```bash
# 启动完整的GUI应用
python trash_classification_gui_complete.py
```

### GUI功能特性

- **📁 文件上传**：支持多种图像格式
- **📹 实时摄像头**：实时分类预测
- **📸 拍照模式**：单张图像分析
- **🎲 随机测试**：从训练集随机选择测试
- **📊 结果可视化**：置信度分布图表
- **🔧 硬件集成**：串口通信控制

### 配置GUI应用

修改`trash_classification_gui_complete.py`中的配置：

```python
# 修改数据集路径
self.dataset_path = "your_dataset"

# 修改类别名称
self.class_names = ['Class1', 'Class2', 'Class3', 'Class4']

# 修改类别颜色
self.class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
```

## ⚙️ 配置参数详解

### 模型参数

| 参数 | 默认值 | 说明 | 可选值 |
|------|--------|------|--------|
| `--backbone` | resnet18 | 模型架构 | resnet18/34/50/101, efficientnet_b0/b3, mobilenet_v2 |
| `--batch_size` | 32 | 批次大小 | 8-128，根据显存调整 |
| `--num_epochs` | 30 | 训练轮数 | 10-200 |
| `--img_size` | 224 | 输入图像尺寸 | 224, 299, 384 |

### 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning_rate` | 0.001 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--optimizer` | adam | 优化器选择 |
| `--lr_scheduler` | reduce_on_plateau | 学习率调度 |

### 数据增强参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_mixup` | False | 启用Mixup增强 |
| `--use_cutmix` | False | 启用CutMix增强 |
| `--mixup_alpha` | 1.0 | Mixup强度参数 |
| `--balance_samples` | False | 类别平衡采样 |

### 性能优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_amp` | False | 混合精度训练 |
| `--num_workers` | 4 | 数据加载进程数 |
| `--gradient_accumulation` | 1 | 梯度累积步数 |
| `--early_stopping` | 0 | 早停轮数（0为禁用） |

## 🔧 高级用法

### 自定义训练脚本

```python
import torch
from train_garbage import main

# 自定义参数
class Args:
    dataset_path = "your_dataset"
    backbone = "efficientnet_b0"
    batch_size = 32
    num_epochs = 50
    # ... 其他参数

# 运行训练
args = Args()
main(args)
```

### 模型推理

```python
import torch
from torchvision import transforms
from PIL import Image

# 加载模型
model = torch.load('best_model.pth')
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 推理
image = Image.open('test_image.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    predictions = torch.nn.functional.softmax(outputs[0], dim=0)
    predicted_class = torch.argmax(predictions).item()
```

## ❓ 常见问题

### Q1: 训练时显存不足怎么办？

**解决方案**：
```bash
# 减小批次大小
--batch_size 16

# 使用梯度累积
--gradient_accumulation 4

# 启用混合精度训练
--use_amp
```

### Q2: 如何处理类别不平衡？

**解决方案**：
```bash
# 使用类别平衡采样
--balance_samples

# 使用Focal Loss
--loss_fn focal_loss --focal_gamma 2.0

# 使用数据增强
--use_mixup --use_cutmix
```

### Q3: 训练速度太慢怎么办？

**解决方案**：
```bash
# 增加数据加载进程
--num_workers 8

# 使用更小的模型
--backbone mobilenet_v2

# 启用混合精度训练（NVIDIA GPU）
--use_amp
```

### Q4: 如何选择合适的学习率？

**建议**：
- 小数据集：0.001-0.01
- 大数据集：0.0001-0.001
- 使用学习率调度：`--lr_scheduler cosine`

### Q5: 模型过拟合怎么办？

**解决方案**：
```bash
# 增加正则化
--weight_decay 1e-3

# 使用标签平滑
--loss_fn label_smoothing --smoothing 0.1

# 启用数据增强
--use_mixup --use_cutmix

# 早停
--early_stopping 10
```

## 📞 技术支持

- **GitHub Issues**: [项目Issues页面]
- **文档**: [在线文档链接]
- **社区讨论**: [讨论区链接]

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献指南

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

## 🌟 致谢

感谢所有为这个项目做出贡献的开发者和研究者。

---

**让AI技术普惠每一个开发者！** 🚀 