# YOLO26 & RT-DETR 通用目标检测训练代码

<p align="center">
    <img src="assets/icon.png" width="110" style="margin-bottom: 0.2;"/>
</p>

<h2 align="center">YOLO 系列 & DETR 系列 通用目标检测训练代码</h2>

<p align="center">
    <a href="https://github.com/Dadalia1917/Yolo-RT-DETR-Train-Code.git">
        <img src="https://img.shields.io/badge/GitHub-Repository-blue" alt="GitHub">
    </a>
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-2.9+-red" alt="PyTorch">
    </a>
    <a href="https://python.org/">
        <img src="https://img.shields.io/badge/Python-3.11-green" alt="Python">
    </a>
    <a href="https://docs.ultralytics.com/models/yolo26/">
        <img src="https://img.shields.io/badge/YOLO26-Latest-orange" alt="YOLO26">
    </a>
</p>

## 📋 项目简介

这是一个通用的目标检测模型训练代码项目，支持 **YOLO 系列** 和 **DETR 系列** 多种先进目标检测模型的训练。项目基于 Ultralytics 框架，提供完整的训练、验证和部署解决方案。

### 支持的模型

| 模型系列 | 版本 | 说明 |
|---------|------|------|
| **YOLO26** ⭐ | n/s/m/l/x | 最新一代 YOLO 模型，**推荐使用** |
| RT-DETR | L/X | 基于 Transformer 的实时检测器 |
| YOLOv5 | n/s/m/l/x | 经典稳定版本，生态成熟 |
| YOLOv8 | n/s/m/l/x | 经典 YOLO 版本 |
| YOLOv11 | n/s/m/l/x | YOLO v11 系列 |
| YOLOv12 | n/s/m/l/x | YOLO v12 系列 |

## ✨ YOLO26 主要特性

YOLO26 是 Ultralytics 推出的最新一代目标检测模型，相比之前版本有显著改进：

- 🚀 **更高精度**: mAP 提升显著，YOLO26n 达到 40.9 mAP
- ⚡ **更快速度**: 推理速度进一步优化，TensorRT 加速仅需 1.7ms
- 🎯 **更少参数**: 参数量和计算量更优，YOLO26n 仅 2.4M 参数
- 🔧 **更易部署**: 支持 ONNX、TensorRT、CoreML 等多种导出格式

### YOLO26 性能对比

| 模型 | 尺寸 | mAP<sup>val</sup> | CPU (ms) | TensorRT (ms) | 参数量 (M) | FLOPs (B) |
|------|------|-------------------|----------|---------------|-----------|-----------|
| YOLO26n | 640 | **40.9** | 38.9 | **1.7** | 2.4 | 5.4 |
| YOLO26s | 640 | **48.6** | 87.2 | 2.5 | 9.5 | 20.7 |
| YOLO26m | 640 | **53.1** | 220.0 | 4.7 | 20.4 | 68.2 |
| YOLO26l | 640 | **55.0** | 286.2 | 6.2 | 24.8 | 86.4 |
| YOLO26x | 640 | **57.5** | 525.8 | 11.8 | 55.7 | 193.9 |

## 🔥 YOLOv5 模型介绍

YOLOv5 是 Ultralytics 推出的经典目标检测模型，以其稳定性和易用性著称，拥有最成熟的社区生态。

### YOLOv5 主要特性

- 🔧 **成熟稳定**: 经过大量实际项目验证，bug 少、文档全
- 📚 **丰富生态**: 社区资源最多，教程、预训练模型、部署方案丰富
- ⚡ **高效部署**: 支持 ONNX、TensorRT、OpenVINO 等多种部署格式
- 🎯 **易于调参**: 配置简单，适合新手快速上手

### YOLOv5 性能对比（COCO 数据集）

| 模型 | 尺寸 | mAP<sup>val</sup> | CPU Speed (ms) | GPU Speed (ms) | 参数量 (M) | FLOPs (B) |
|------|------|-------------------|----------------|----------------|------------|-----------|
| YOLOv5n | 640 | 28.0 | 45.0 | 1.3 | 1.9 | 4.5 |
| YOLOv5s | 640 | 37.4 | 98.0 | 2.1 | 7.2 | 16.5 |
| YOLOv5m | 640 | 45.4 | 224.0 | 3.8 | 21.2 | 49.0 |
| YOLOv5l | 640 | 49.0 | 430.0 | 5.4 | 46.5 | 109.1 |
| YOLOv5x | 640 | 50.7 | 766.0 | 8.2 | 86.7 | 205.7 |

### YOLOv5 适用场景

- 🏭 **工业部署**: 需要高稳定性和成熟部署方案的生产环境
- 📱 **边缘设备**: 需要轻量级模型部署到嵌入式设备
- 🎓 **学习入门**: 目标检测初学者快速入门
- 🔬 **论文复现**: 大量论文基于 YOLOv5 进行改进

## 🛠️ 环境安装

### 1. 创建 Python 环境

```bash
# 切换到国内镜像源（推荐）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 创建并激活 conda 虚拟环境
conda create -n yolo python=3.11
conda activate yolo
```

### 2. 安装核心依赖

```bash
# 安装 Ultralytics
pip install -U ultralytics

# 安装 CUDA 12.8 版本的 PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3. 解决 pynvml 警告（可选）

如果看到 `pynvml package is deprecated` 警告，可以安装替代包：

```bash
pip uninstall pynvml -y
pip install nvidia-ml-py
```

### 4. 安装其他依赖

```bash
# 数据增强库
pip install -U albumentations

# UI 界面库
pip install pyqt5

# 其他依赖
pip install flask flask-socketio
```

## 🚀 快速开始

### 1. 训练 YOLO26 模型（推荐）

```bash
python train_v26.py
```

训练脚本会自动下载 `yolo26n.pt` 预训练权重并开始训练。

### 2. 训练其他模型

```bash
# RT-DETR 训练
python train_rtdetr.py

# YOLOv5 训练（经典稳定版本）
python train_v5.py

# YOLOv8 训练
python train_v8.py

# YOLOv11 训练
python train_v11.py

# YOLOv12 训练
python train_v12.py

# YOLOv13 训练
python train_v13.py
```

### 3. GPU 选择

训练脚本支持指定使用的 GPU。在脚本开头修改 `CUDA_VISIBLE_DEVICES`：

```python
import os
# 使用 GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 或使用 GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
```

使用 `check_gpu.py` 工具查看系统中的 GPU 信息：

```bash
python check_gpu.py
```

## 📁 项目结构

```
├── assets/                    # 项目资源文件
├── datasets/                  # 数据集目录
│   └── Data/                  # 训练数据
│       ├── data.yaml          # 数据集配置
│       ├── train/             # 训练集
│       └── val/               # 验证集
├── ultralytics/               # 核心训练库
├── UIProgram/                 # PyQt5 图形界面
├── runs/                      # 训练结果输出
├── train_v26.py               # YOLO26 训练脚本 ⭐
├── train_rtdetr.py            # RT-DETR 训练脚本
├── train_v5.py                # YOLOv5 训练脚本（经典版）
├── train_v8.py                # YOLOv8 训练脚本
├── train_v11.py               # YOLOv11 训练脚本
├── train_v12.py               # YOLOv12 训练脚本
├── train_v13.py               # YOLOv13 训练脚本
└── README.md                  # 项目说明
```

## 🔧 配置说明

### 训练参数

可在训练脚本中修改以下参数：

```python
results = model.train(
    data='datasets/Data/data.yaml',  # 数据集配置
    epochs=300,                       # 训练轮数
    imgsz=640,                        # 输入图像尺寸
    batch=16,                         # 批次大小
    optimizer='SGD',                  # 优化器（推荐 SGD）
    amp=True,                         # 自动混合精度训练
    device=device                       # 使用的 GPU
)
```

### 训练优化

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| optimizer | `SGD` | SGD 在目标检测任务中通常比 Adam 获得更高精度 |
| amp | `True` | 自动混合精度可加速训练 1.5-2 倍，节省 40-50% 显存 |
| batch | 16 | 根据 GPU 显存调整，16GB 显存推荐 16 |

## 📊 模型导出

训练完成后，可以导出模型用于部署：

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train_v26/weights/best.pt')

# 导出为 ONNX
model.export(format='onnx')

# 导出为 TensorRT (需要 TensorRT 环境)
model.export(format='engine')
```

## 🔗 相关链接

- [YOLO26 官方文档](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [RT-DETR 论文](https://arxiv.org/abs/2304.08069)

## 📝 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

## 📞 联系方式

- GitHub Issues: [提交问题](https://github.com/Dadalia1917/Yolo-RT-DETR-Train-Code/issues)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
