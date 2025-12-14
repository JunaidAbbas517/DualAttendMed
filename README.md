# DualAttendMed: A Coarse-to-Fine Dual-Stage Attention Framework for Interpretable Disease Localization and Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Code Structure](#code-structure)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

**DualAttendMed** is a novel dual-stage attention framework designed to enhance disease localization and classification in medical images. The framework addresses critical challenges in medical image analysis, including precise localization, imbalanced class distributions, and complex disease patterns, while providing interpretable visual explanations for clinical decision-making.

### Highlights

- ✅ **State-of-the-art performance** on retinal fundus datasets (APTOS, DDR, Messidor-2)
- ✅ **Interpretable attention maps** rated 4.7/5 by clinical experts
- ✅ **Coarse-to-fine attention strategy** for improved localization
- ✅ **Attention-Assisted Data Augmentation (AADA)** for robust training
- ✅ **Composite loss function** balancing classification and localization

## 📄 Abstract

Accurately diagnosing medical images is crucial in healthcare, but challenges, including precise localization, imbalanced class distributions, and complex disease patterns, significantly limit diagnostic accuracy. Existing methods often prioritize classification accuracy by deriving global features through backbone networks, such as CNNs, and feeding them into classifiers. However, these approaches frequently exhibit deficiencies in interpretability and transparency, which are essential for medical applications. To address these limitations, we introduce **DualAttendMed**, a novel dual-stage attention framework designed to enhance disease localization and classification in medical images. Utilizing ResNet-152 for initial feature extraction, channel attention refines these features to emphasize disease-relevant regions, while attention-assisted data augmentation employs assisted cropping and dropping to generate interpretable attention maps, exploring both prominent and subtle features through a coarse-to-fine attention strategy, improving both global context and fine-grained localization. A composite loss function balances classification accuracy, attention alignment, and feature diversity, ensuring precise disease localization. Validated through comprehensive experiments on three retinal fundus datasets (APTOS, DDR, and Messidor-2), DualAttendMed achieves state-of-the-art classification accuracies of **92.50%**, **87.10%**, and **88.70%**, respectively, along with IoU values of **0.85**, **0.80**, and **0.75**, demonstrating superior performance. Its visual explanations enhance deep learning clinical confidence, supporting early detection, precise diagnosis, and seamless integration into diagnostic workflows. By bridging deep learning and clinical practice, DualAttendMed provides a robust and accurate method for early disease identification, significantly improving healthcare outcomes and patient care.

## ✨ Key Features

### 1. **Dual-Stage Attention Mechanism**
   - **Coarse Stage**: Captures global context and prominent disease regions
   - **Fine Stage**: Refines attention to subtle and fine-grained features
   - Iterative refinement for improved localization accuracy

### 2. **Channel Attention Module**
   - Refines ResNet-152 features to emphasize disease-relevant channels
   - Enhances discriminative power for medical image classification

### 3. **Attention-Assisted Data Augmentation (AADA)**
   - **Attention-assisted cropping**: Focuses on critical regions
   - **Attention-assisted dropping**: Suppresses non-informative features
   - Addresses class imbalance and improves generalization

### 4. **Bilinear Attention Pooling (BAP)**
   - Integrates spatial and channel-wise attention features
   - Generates discriminative feature representations

### 5. **Composite Loss Function**
   - **Classification Loss**: Ensures accurate disease classification
   - **Attention Alignment Loss**: Aligns attention maps for consistency
   - **Diversity Loss**: Encourages diverse attention patterns
   - Balanced weighting: λ₁=1.0, λ₂=0.5, λ₃=0.1

### 6. **Interpretable Visualizations**
   - Generates attention heatmaps for clinical interpretation
   - Provides CAM-based localization maps
   - Supports clinical decision-making with transparent explanations

## 🏗️ Architecture

```
Input Image (224×224)
    ↓
ResNet-152 Backbone (Feature Extraction)
    ↓
Channel Attention Module (Feature Refinement)
    ↓
Attention Maps Generation (M=32 attention maps)
    ↓
Bilinear Attention Pooling (BAP)
    ↓
Classification Head
    ↓
Output: Class Prediction + Attention Maps
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM (for training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/JunaidAbbas517/DualAttendMed.git
cd DualAttendMed
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install requirements

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🎮 Quick Start

### 1. Dataset Preparation

Organize your dataset in the following structure:

```
data/
├── train_images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── train.csv          # Format: filename,label
├── test_images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── test.csv           # Format: filename,label
```

**CSV Format Example:**
```csv
filename,label
image1.png,0
image2.png,1
image3.png,2
```

### 2. Configure Training Parameters

Edit `config_distributed.py`:

```python
GPU = '0'                    # GPU ID
epochs = 40                  # Training epochs
batch_size = 32              # Batch size
learning_rate = 1e-3         # Initial learning rate
image_size = (224, 224)      # Input image size
num_attentions = 32          # Number of attention maps
tag = 'CT'                   # Dataset tag
save_dir = './checkpoints/'  # Model save directory
```

### 3. Train the Model

```bash
python train_distributed.py
```

Or use the provided shell script:

```bash
bash train_distributed.sh
```

### 4. Run Inference

Edit `config_infer.py`:

```python
ckpt = './checkpoints/model_bestacc.pth'  # Path to trained model
tag = 'CT'                                 # Dataset tag
use_cam_iou = True                         # Enable IoU evaluation
```

Run inference:

```bash
python infer.py
```


## 📁 Code Structure

```
DualAttendMed/
├── models/
│   ├── __init__.py          # Model exports
│   ├── cal.py               # WSDAN_CAL (DualAttendMed) model
│   ├── resnet.py            # ResNet-152 backbone
│   └── blocks.py            # CBAM layer
├── datasets/
│   ├── __init__.py          # Dataset factory
│   └── mydataset.py         # Custom dataset loader
├── config_distributed.py    # Training configuration
├── config_infer.py          # Inference configuration
├── train_distributed.py     # Training script
├── infer.py                 # Inference and evaluation script
├── cam_generator.py         # CAM generation for IoU evaluation
├── grand_cam_utils.py       # GradCAM utilities
├── utils.py                 # Loss functions and utilities
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

### Key Components

- **`models/cal.py`**: Main DualAttendMed model with dual-stage attention
- **`utils.py`**: Loss functions (AttentionAlignmentLoss, DiversityLoss, etc.)
- **`cam_generator.py`**: CAM-based pseudo-ground truth generation for IoU
- **`train_distributed.py`**: Training loop with composite loss
- **`infer.py`**: Inference with attention visualization and IoU evaluation

## 🔬 Training Details

### Hyperparameters

- **Backbone**: ResNet-152 (pretrained on ImageNet)
- **Number of Attention Maps**: 32 (M=32)
- **Input Size**: 224×224 pixels
- **Optimizer**: Adam or SGD (configurable)
- **Learning Rate**: 1e-3 (with decay)
- **Loss Weights**: λ₁=1.0, λ₂=0.5, λ₃=0.1

### Training Process

1. **Forward Pass**: Extract features → Generate attention maps → Classify
2. **Loss Computation**: Composite loss (classification + attention + diversity)
3. **Backward Pass**: Update model parameters
4. **Validation**: Evaluate on validation set with IoU calculation

### Monitoring Training

Training logs are saved to `save_dir/log_name`. Monitor:
- Training/validation accuracy
- Loss components (classification, attention, diversity)
- IoU values (if CAM masks enabled)

## 🔍 Inference and Visualization

### Generate Attention Maps

```python
from models import WSDAN_CAL
import torch

# Load model
model = WSDAN_CAL(num_classes=8, M=32, net='resnet152')
checkpoint = torch.load('checkpoints/model_bestacc.pth')
model.load_state_dict(checkpoint['state_dict'])

# Get predictions and attention maps
predictions, attention_maps = model(image_batch)
```

### Visualize Attention Heatmaps

The inference script automatically generates attention visualizations when `visual_path` is set in `config_infer.py`.

### Evaluate IoU

Set `use_cam_iou = True` in `config_infer.py` to enable IoU evaluation using CAM-based pseudo-ground truth.


## 👥 Authors

- **Junaid Abbas** - School of Big Data and Software Engineering, Chongqing University
- **Danyal Badar Soomro** - College of Computer Science, Chongqing University
- **Shanshan Huang** - School of Big Data and Software Engineering, Chongqing University
- **Li Liu** - School of Big Data and Software Engineering, Chongqing University

## 🙏 Acknowledgments

- Chongqing University for research support and resources
- The open-source community for excellent tools and libraries

## 📧 Contact

For questions, collaborations, or feedback:

- **Junaid Abbas**: 
- **Danyal Badar Soomro**:
- **Shanshan Huang**: 
- **Li Liu**: 

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

**Note**: This repository contains the official implementation of DualAttendMed. The paper is currently under review. For the latest updates and pre-trained models, please check the [releases](https://github.com/JunaidAbbas517/DualAttendMed/releases) section.

---

<div align="center">
  <p>Made with ❤️ by the DualAttendMed Team</p>
  <p>Chongqing University, 2025</p>
</div>

