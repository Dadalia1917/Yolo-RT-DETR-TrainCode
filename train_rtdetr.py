# coding:utf-8
import torch
from ultralytics import RTDETR  # 确保 RTDETR 模型正确导入

# 模型配置文件路径
model_yaml_path = "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml"  # RT-DETR 配置文件
# 数据集配置文件路径
data_yaml_path = 'datasets/Data/data.yaml'
# 预训练模型文件路径
pre_model_name = 'rtdetr-l.pt'  # RT-DETR 预训练模型

if __name__ == '__main__':
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建RT-DETR模型并加载预训练权重
    model = RTDETR(model_yaml_path).to(device)  # 先创建模型并移动到设备
    model.load(pre_model_name)  # 加载预训练模型权重

    # 训练模型
   results = model.train(data=data_yaml_path,
                          epochs=300,
                          batch=8,  # 批次大小
                          project='runs/detect',  # 指定保存到当前项目的runs/detect目录
                          name='train_rtdetr',
                          workers=8,  # 调整数据加载线程数
                          optimizer='SGD',  # 使用SGD优化器，通常比Adam获得更好的最终精度
                          amp=True)  # 启用自动混合精度训练，加速训练并节省显存
