# coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from ultralytics import YOLO

# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/13/yolo13.yaml"
# 数据集配置文件
data_yaml_path = 'datasets/Data/data.yaml'
# 预训练模型
pre_model_name = 'yolov13n.pt'

if __name__ == '__main__':
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 注意：通常cuda:0是第一块卡，cuda:1是第二块卡
    # 可以通过 nvidia-smi 确认GPU索引
    device = '0'  # Ultralytics框架接受字符串形式的设备ID

    # 直接加载预训练模型，而不是先加载yaml再加载权重
    model = YOLO(pre_model_name)

    # 训练模型
    results = model.train(data=data_yaml_path,
                          epochs=300,
                          imgsz=640,
                          batch=16,  # 批次大小
                          project='runs/detect',  # 指定保存到当前项目的runs/detect目录
                          name='train_v13',
                          workers=10,  # 调整数据加载线程数，匹配14核CPU
                          optimizer='SGD',  # 使用SGD优化器，通常比Adam获得更好的最终精度
                          amp=True,  # 启用自动混合精度训练，加速训练并节省显存
                          device=device)  # 指定使用的GPU设备