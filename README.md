# YOLO & RT-DETR 通用训练代码

<p align="center">
    <img src="assets/icon.png" width="110" style="margin-bottom: 0.2;"/>
</p>

<h2 align="center">YOLO & RT-DETR 通用目标检测训练代码</h2>

<p align="center">
    <a href="https://github.com/Dadalia1917/Yolo-RT-DETR-Train-Code.git">
        <img src="https://img.shields.io/badge/GitHub-Repository-blue" alt="GitHub">
    </a>
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-Framework-red" alt="PyTorch">
    </a>
    <a href="https://python.org/">
        <img src="https://img.shields.io/badge/Python-3.11-green" alt="Python">
  </a>
</p>

## 📋 项目简介

这是一个通用的目标检测模型训练代码项目，支持多种先进的目标检测模型训练，包括：

- **RT-DETR** (Real-Time Detection with Vision Transformers)
- **YOLOv8** 系列模型
- **YOLOv9-v13** 最新系列模型
- **YOLO11-12** 最新版本

项目提供了完整的训练、验证、测试和部署解决方案，适用于各种目标检测任务。

## ✨ 主要特性

- 🚀 **多模型支持**: 支持RT-DETR和YOLO系列（v8-v13）的训练
- 🎯 **通用适配**: 适用于各种目标检测任务和数据集
- 🖥️ **可视化界面**: 提供PyQt5图形界面，方便操作
- 🐳 **Docker支持**: 提供多种Docker配置，支持不同硬件平台
- 📊 **实时监控**: 训练过程可视化和日志记录
- 🔧 **灵活配置**: 支持多种配置文件和训练策略
- 📈 **性能分析**: 提供详细的性能评估工具

## 🛠️ 环境安装

### 1. 配置Python环境

首先切换到国内镜像源（推荐）：
```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

创建并激活conda虚拟环境：
```bash
conda create -n yolo python=3.11
conda activate yolo
```

### 2. 安装核心依赖

安装Ultralytics：
```bash
pip install -U ultralytics
```

### 3. 安装PyTorch（CUDA 12.8）

卸载现有PyTorch版本：
```bash
pip uninstall torch
pip uninstall torchvision
```

安装CUDA 12.8版本的PyTorch：
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 4. 安装Flash Attention

```bash
pip install D:\download\flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl
```

### 5. 安装其他依赖

```bash
# 安装指定版本的torch和xformers
pip install torch==2.8.0 xformers

# 安装数据增强库
pip install -U albumentations

# 安装机器学习相关库
pip install huggingface_hub datasets

# 安装UI界面库
pip install pyqt5

# 安装Web服务相关库
pip install flask flask-socketio openai

# 安装数据库相关库
pip install sqlalchemy flask_bcrypt flask_login
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Dadalia1917/Yolo-RT-DETR-Train-Code.git
cd Yolo-RT-DETR-Train-Code
```

### 2. 准备数据集

将您的数据集放在 `datasets/Data/` 目录下，确保包含以下结构：
```
datasets/
├── Data/
│   ├── data.yaml          # 数据集配置文件
│   ├── train/            # 训练图像
│   │   ├── images/
│   │   └── labels/
│   └── val/              # 验证图像
│       ├── images/
│       └── labels/
```

### 3. 训练模型

#### 使用命令行训练

**训练RT-DETR：**
```bash
python MainProgram_RTDETR.py
```

**训练不同版本的YOLO：**
```bash
# YOLOv11训练
python train_v11.py

# YOLOv12训练
python train_v12.py

# YOLOv13训练
python train_v13.py
```

#### 使用图形界面

启动PyQt5图形界面：
```bash
cd UIProgram
python UiMain.py
```

### 4. 模型预测

使用训练好的模型进行预测：
```bash
python prediction.py --model path/to/your/model.pt --source path/to/your/image
```

## 📁 项目结构

```
├── assets/                    # 项目资源文件
├── datasets/                  # 数据集目录
│   └── Data/                 # 训练数据
├── UIProgram/                # PyQt5图形界面
├── docker/                   # Docker配置文件
├── docs/                     # 文档
├── examples/                 # 示例代码
├── ultralytics/              # 核心训练库
├── Config1.py               # 配置文件1
├── Config2.py               # 配置文件2
├── MainProgram_RTDETR.py    # RT-DETR主程序
├── MainProgram_yolo.py      # YOLO主程序
├── train_v11.py             # YOLOv11训练脚本
├── train_v12.py             # YOLOv12训练脚本
├── train_v13.py             # YOLOv13训练脚本
├── train.py                 # 通用训练脚本
├── prediction.py            # 预测脚本
├── detect_tools.py          # 检测工具
└── requirements.txt         # 依赖列表
```

## 🔧 配置说明

### 模型配置

项目支持以下预训练模型：

**YOLO系列：**
- YOLOv8: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- YOLOv10: `yolov10n.pt`, `yolov10s.pt`, `yolov10m.pt`, `yolov10l.pt`, `yolov10x.pt`
- YOLOv11: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- YOLOv12: `yolo12n.pt`, `yolo12s.pt`, `yolo12m.pt`, `yolo12l.pt`, `yolo12x.pt`
- YOLOv13: `yolov13n.pt`, `yolov13s.pt`, `yolov13l.pt`, `yolov13x.pt`

**RT-DETR系列：**
- RT-DETR-L: `rtdetr-l.pt`
- RT-DETR-X: `rtdetr-x.pt`

### 训练参数

可以在配置文件中修改以下主要参数：

```python
# 基本训练参数
epochs = 300              # 训练轮数
batch_size = 16          # 批次大小
learning_rate = 0.01     # 学习率
img_size = 640           # 输入图像尺寸

# 数据增强参数
mosaic = 1.0             # Mosaic数据增强
mixup = 0.0              # Mixup数据增强
copy_paste = 0.1         # Copy-Paste数据增强
```

## 🐳 Docker部署

项目提供了多种Docker配置：

```bash
# 基础CPU版本
docker build -f docker/Dockerfile-cpu -t yolo-cpu .

# GPU版本
docker build -f docker/Dockerfile -t yolo-gpu .

# Jetson版本
docker build -f docker/Dockerfile-jetson -t yolo-jetson .
```

## 📊 模型性能

### YOLO系列性能对比

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 参数量(M) | FLOPs(G) | 推理速度(ms) |
|------|---------|--------------|-----------|----------|-------------|
| YOLOv8n | 52.6 | 37.4 | 3.2 | 8.7 | 1.77 |
| YOLOv11n | 54.2 | 38.6 | 2.6 | 6.5 | 1.53 |
| YOLOv12n | 56.0 | 40.1 | 2.6 | 6.5 | 1.83 |
| YOLOv13n | 57.8 | 41.6 | 2.5 | 6.4 | 1.97 |

### RT-DETR性能

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 参数量(M) | FLOPs(G) | 推理速度(ms) |
|------|---------|--------------|-----------|----------|-------------|
| RT-DETR-R50 | 71.3 | 53.1 | 42.0 | 136.0 | 6.93 |
| RT-DETR-R101 | 72.7 | 54.3 | 76.0 | 259.0 | 13.51 |

## 🔗 相关链接

- [Ultralytics官方文档](https://docs.ultralytics.com/)
- [RT-DETR论文](https://arxiv.org/abs/2304.08069)
- [YOLOv8论文](https://arxiv.org/abs/2305.09972)
- [项目GitHub地址](https://github.com/Dadalia1917/Yolo-RT-DETR-Train-Code.git)

## 📝 许可证

本项目基于 [MIT许可证](LICENSE) 开源。

## 🤝 贡献

欢迎提交Issue和Pull Request来帮助改进项目！

## 📞 联系方式

如有问题，请通过以下方式联系：

- GitHub Issues: [提交问题](https://github.com/Dadalia1917/Yolo-RT-DETR-Train-Code/issues)
- 项目作者: 张金翔

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
