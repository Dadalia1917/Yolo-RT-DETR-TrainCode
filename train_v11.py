# coding:utf-8
import torch
from ultralytics import YOLO

# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/v11/yolo11x.yaml"
# 数据集配置文件
data_yaml_path = 'datasets/Data/data.yaml'
# 预训练模型
pre_model_name = 'yolo11x.pt'

if __name__ == '__main__':
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练模型到指定设备
    model = YOLO(model_yaml_path).load(pre_model_name).to(device)

    # 训练模型
    results = model.train(data=data_yaml_path,
                          epochs=150,
                          batch=12,  # 批次大小
                          name='train_v11',
                          workers=16,  # 调整数据加载线程数
                          optimizer='SGD')  # 使用SGD优化器
